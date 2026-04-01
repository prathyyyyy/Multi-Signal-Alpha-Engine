# Databricks notebook source
# MAGIC %pip install requests beautifulsoup4 lxml aiohttp nest_asyncio --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timezone
import pandas as pd
import requests
import time
import re
import random
import asyncio
import aiohttp
import nest_asyncio
import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore

# COMMAND ----------

nest_asyncio.apply()

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

STORAGE_ACCOUNT = "multisignalalphaeng"
CONTAINER       = "quant-lakehouse"
ADLS_KEY        = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")
spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH       = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
CHECKPOINT_PATH = "/tmp/sec_filings_checkpoint.parquet"

print("Config loaded ✓")
print(f"Checkpoint path: {CHECKPOINT_PATH}")

# COMMAND ----------

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# SEC EDGAR requires a user-agent header — use your email
SEC_HEADERS = {
    "User-Agent": "Quant Research Project prathyleetcode@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

print("Config loaded ✓")

# COMMAND ----------

class BronzeSECFilingsFinal:
    """
    Bronze SEC EDGAR ingestion — Final Version
    -----------------------------------------------
    Phase 1 : aiohttp async  — fetch all filing indexes (50 concurrent)
    Phase 2 : ThreadPool     — fetch text via confirmed URL format
    Checkpoint: saves parquet right after ingestion completes
    Confirmed URL: /Archives/edgar/data/{cik}/{acc}/{acc_fmt}-index.htm
    Est. time : ~2.5 hours total
    """

    FILING_TYPES   = {"10-K", "10-Q", "8-K"}
    TEXT_FROM_YEAR = 2015
    MAX_CONCURRENT = 50
    N_TEXT_THREADS = 6

    SECTIONS = {
        "10-K": ["item 1a", "item 7"],
        "10-Q": ["item 2", "item 3"],
        "8-K" : ["item 2", "item 8"],
    }

    HEADERS = {
        "User-Agent"     : "Quant Research Project prathyleetcode@gmail.com",
        "Accept"         : "text/html, application/json, */*",
        "Accept-Encoding": "gzip, deflate",
        "Connection"     : "keep-alive",
    }

    def __init__(self, spark, base_path, tickers,
                 checkpoint_path="/tmp/sec_filings_checkpoint.parquet"):
        self.spark           = spark
        self.path            = f"{base_path}/bronze/delta/sec_filings"
        self.tickers         = tickers
        self.checkpoint_path = checkpoint_path
        self.failed          = []
        self._lock           = Lock()
        self._semaphore      = Semaphore(self.N_TEXT_THREADS)

        print("BronzeSECFilingsFinal ✓")
        print(f"  Tickers        : {len(tickers)}")
        print(f"  Full text from : {self.TEXT_FROM_YEAR}")
        print(f"  Async index    : {self.MAX_CONCURRENT} concurrent")
        print(f"  Text threads   : {self.N_TEXT_THREADS}")
        print(f"  Checkpoint     : {checkpoint_path}")
        print(f"  Est. time      : ~2.5 hours")

    # ------------------------------------------------------------------ #
    #  CIK Map
    # ------------------------------------------------------------------ #
    def _build_cik_map(self) -> dict:
        print("\nFetching CIK map from SEC EDGAR...")
        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=self.HEADERS,
                timeout=30
            )
            if r.status_code == 200 and len(r.content) > 100:
                data    = r.json()
                cik_map = {
                    v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                    for v in data.values()
                }
                print(f"  ✓ {len(cik_map):,} companies loaded from EDGAR")
                return cik_map
        except Exception as e:
            print(f"  ✗ Live fetch failed: {e}")
        print("  Using hardcoded CIK map fallback")
        return self._hardcoded_cik_map()

    def _hardcoded_cik_map(self) -> dict:
        return {
            "AAPL":"0000320193","MSFT":"0000789019","NVDA":"0001045810",
            "AMZN":"0001018724","GOOGL":"0001652044","GOOG":"0001652044",
            "META":"0001326801","TSLA":"0001318605","BRK-B":"0001067983",
            "UNH":"0000072971","LLY":"0000059478","JPM":"0000019617",
            "V":"0001403161","XOM":"0000034088","MA":"0001141391",
            "AVGO":"0001730168","PG":"0000080424","JNJ":"0000200406",
            "HD":"0000354950","MRK":"0000310158","CVX":"0000093410",
            "ABBV":"0001551152","BAC":"0000070858","COST":"0000909832",
            "PEP":"0000077476","KO":"0000021344","WMT":"0000104169",
            "CSCO":"0000858877","MCD":"0000063908","CRM":"0001108524",
            "ACN":"0001467373","TMO":"0000097476","ABT":"0000001800",
            "ADBE":"0000796343","DHR":"0000313616","WFC":"0000072971",
            "PM":"0001413329","MS":"0000895421","GS":"0000886982",
            "INTU":"0000896878","AMD":"0000002488","ISRG":"0001035267",
            "CAT":"0000018230","RTX":"0000101830","SPGI":"0000064040",
            "LOW":"0000060667","AMGN":"0000820081","VRTX":"0000875320",
            "NOW":"0001373715","PLD":"0001045609","BKNG":"0001075531",
            "AXP":"0000004962","UBER":"0001543151","DE":"0000315189",
            "SYK":"0000310764","GILD":"0000882095","ADI":"0000006951",
            "REGN":"0000872589","PGR":"0000080661","ETN":"0001551182",
            "BSX":"0000885725","MDLZ":"0001103982","MU":"0000723125",
            "KLAC":"0000319201","CB":"0000021175","SO":"0000092122",
            "DUK":"0001326160","NEE":"0000753308","GE":"0000040533",
            "LRCX":"0000707549","CME":"0001156375","PANW":"0001327567",
            "CRWD":"0001535527","SNPS":"0000883241","CDNS":"0000813672",
            "TJX":"0000109198","USB":"0000036104","PNC":"0000713676",
            "NOC":"0000202058","AON":"0000315293","ZTS":"0001555280",
            "ITW":"0000049826","EMR":"0000032604","F":"0000037996",
            "GM":"0001467858","FCX":"0000831259","NSC":"0000702165",
            "UNP":"0000100885","FDX":"0000230569","UPS":"0001090727",
            "LIN":"0001707092","APD":"0000002969","ECL":"0000031462",
            "NKE":"0000320187","TGT":"0000027419","MCO":"0001059556",
            "ICE":"0001571949","HUM":"0000049071","CI":"0001739940",
            "CVS":"0000064803","QCOM":"0000804328","ORCL":"0001341439",
            "IBM":"0000051143","INTC":"0000050863","HON":"0000773840",
            "MMM":"0000066740","BA":"0000012927","LMT":"0000936468",
            "GD":"0000040533","AIG":"0000005272","MET":"0001099590",
            "PRU":"0001137774","ALL":"0000899051","TRV":"0000086312",
            "PFE":"0000078003","BIIB":"0000875045","MRNA":"0001682852",
            "BMY":"0000014272","ELV":"0001156039","HCA":"0000860730",
            "ADM":"0000007084","NEM":"0001164180","OXY":"0000797468",
            "HAL":"0000045012","SLB":"0000087347","COP":"0001163165",
            "EOG":"0000821189","PSX":"0001534992","VLO":"0001035002",
            "MPC":"0001510295","DIS":"0001744489","CMCSA":"0001166691",
            "NFLX":"0001065280","CHTR":"0001091907","T":"0000732717",
            "VZ":"0000732712","TMUS":"0001283699","DXCM":"0001093557",
            "IDXX":"0000874716","A":"0001090872","ALGN":"0001097149",
            "EW":"0001099800","ZBH":"0001136893","STE":"0000093676",
            "RMD":"0000943819","COO":"0000723254","HSIC":"0001000228",
            "IQV":"0001478930","CCL":"0000723254","RCL":"0000884887",
            "NCLH":"0001513761","MAR":"0001048286","HLT":"0001585389",
            "WYNN":"0001174922","MGM":"0000789570","NDAQ":"0001120193",
            "CBOE":"0001374310","MSCI":"0001408198","FDS":"0000040570",
            "VRSK":"0001442145","BR":"0001383312","FIS":"0000798354",
            "FISV":"0000798354","GPN":"0001123360","KEY":"0000091576",
            "RF":"0001281761","HBAN":"0000049196","CFG":"0001378946",
            "MTB":"0000036270","FITB":"0000035527","WBA":"0001339369",
            "KR":"0000056873","MKC":"0000063754","HRL":"0000048102",
            "CAG":"0000023217","GIS":"0000040570","K":"0000055529",
            "HSY":"0000047111","MNST":"0000865752","KDP":"0001418135",
            "STZ":"0000016918","EL":"0001001250","CL":"0000021665",
            "CHD":"0000313927","CLX":"0000021076","BK":"0000009626",
            "NTRS":"0000073124","STT":"0000093751","PFG":"0000945114",
            "AFL":"0000004977","CINF":"0000020286","AJG":"0000354963",
            "MMC":"0000062234","HIG":"0000086312","SNA":"0000091440",
            "PH":"0000076334","DOV":"0000029905","ROK":"0001024478",
            "AME":"0001037049","GWW":"0000277283","FAST":"0000815556",
            "MSI":"0000068505","KEYS":"0001601712","TDY":"0000096289",
            "PAYC":"0001590714","PAYX":"0000723254","ADP":"0000012927",
            "WM":"0000823768","RSG":"0001060349","CTAS":"0000723254",
            "ODFL":"0000878927","JBHT":"0000049216","NSC":"0000702165",
            "UNP":"0000100885","CSX":"0000277948","CP":"0000016875",
        }

    # ------------------------------------------------------------------ #
    #  Phase 1 — Async index fetch
    # ------------------------------------------------------------------ #
    async def _fetch_index_async(self, session, semaphore,
                                  ticker, cik) -> list:
        async with semaphore:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as r:
                    if r.status != 200:
                        return []

                    data   = await r.json(content_type=None)
                    recent = data.get("filings", {}).get("recent", {})

                    forms        = recent.get("form", [])
                    dates        = recent.get("filingDate", [])
                    accessions   = recent.get("accessionNumber", [])
                    primary_docs = recent.get("primaryDocument", [])

                    filings = []
                    for form, date, acc, doc in zip(
                        forms, dates, accessions, primary_docs
                    ):
                        if form in self.FILING_TYPES:
                            filings.append({
                                "ticker"       : ticker,
                                "cik"          : cik,
                                "filing_type"  : form,
                                "filing_date"  : date,
                                "year"         : int(date[:4]),
                                "month"        : int(date[5:7]),
                                "accession"    : acc.replace("-", ""),
                                "accession_fmt": acc,
                                "doc"          : doc,
                            })

                    # Older filings beyond recent 1000
                    for f in data.get("filings", {}).get("files", []):
                        old_url = (
                            f"https://data.sec.gov/submissions/{f['name']}"
                        )
                        try:
                            async with session.get(
                                old_url,
                                timeout=aiohttp.ClientTimeout(total=15)
                            ) as old_r:
                                if old_r.status == 200:
                                    old_data = await old_r.json(
                                        content_type=None
                                    )
                                    for form, date, acc, doc in zip(
                                        old_data.get("form", []),
                                        old_data.get("filingDate", []),
                                        old_data.get("accessionNumber",[]),
                                        old_data.get("primaryDocument",[])
                                    ):
                                        if form in self.FILING_TYPES:
                                            filings.append({
                                                "ticker"       : ticker,
                                                "cik"          : cik,
                                                "filing_type"  : form,
                                                "filing_date"  : date,
                                                "year"         : int(date[:4]),
                                                "month"        : int(date[5:7]),
                                                "accession"    : acc.replace("-",""),
                                                "accession_fmt": acc,
                                                "doc"          : doc,
                                            })
                        except Exception:
                            continue

                    return filings

            except Exception as e:
                with self._lock:
                    self.failed.append((ticker, "index", str(e)))
                return []

    async def _fetch_all_indexes_async(self, ticker_cik_pairs) -> list:
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        connector = aiohttp.TCPConnector(
            limit          = self.MAX_CONCURRENT,
            limit_per_host = 10,
            ttl_dns_cache  = 300
        )
        async with aiohttp.ClientSession(
            headers   = self.HEADERS,
            connector = connector
        ) as session:
            tasks = [
                self._fetch_index_async(session, semaphore, ticker, cik)
                for ticker, cik in ticker_cik_pairs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_filings = []
        for r in results:
            if isinstance(r, list):
                all_filings.extend(r)
        return all_filings

    # ------------------------------------------------------------------ #
    #  Phase 2 — Text fetch
    # ------------------------------------------------------------------ #
    def _extract_sections(self, text: str, filing_type: str) -> str:
        text_lower = text.lower()
        extracted  = []
        for section in self.SECTIONS.get(filing_type, []):
            idx = text_lower.find(section)
            if idx == -1:
                continue
            chunk = re.sub(r'\s+', ' ', text[idx: idx+5000]).strip()
            extracted.append(f"[{section.upper()}] {chunk}")
        if extracted:
            return " | ".join(extracted)[:15000]
        return re.sub(r'\s+', ' ', text[:5000]).strip()

    def _parse_content(self, content: bytes, url: str) -> str:
        try:
            if url.endswith(".txt"):
                return content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "lxml")
            for tag in soup(["table","script","style","header","footer"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return ""

    def _fetch_text(self, filing: dict) -> dict:
        """
        Two-method approach:
        Method 1: Direct doc URL using stored filename (fastest)
        Method 2: Parse index.htm to find doc link (fallback)
        """
        cik         = filing["cik"]
        acc         = filing["accession"]        # no dashes
        acc_fmt     = filing["accession_fmt"]    # with dashes
        doc         = filing["doc"]              # primary doc filename
        filing_type = filing["filing_type"]
        cik_int     = int(cik)

        empty = {
            "accession"  : acc,
            "filing_text": "",
            "has_text"   : False
        }

        with self._semaphore:
            # ── Method 1: Direct doc URL ────────────────────────────────
            if doc and doc.lower().endswith((".htm",".html",".txt")):
                try:
                    time.sleep(0.15 + random.uniform(0, 0.1))
                    url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_int}/{acc}/{doc}"
                    )
                    r = requests.get(url, headers=self.HEADERS, timeout=20)

                    if r.status_code == 429:
                        time.sleep(60)
                        r = requests.get(
                            url, headers=self.HEADERS, timeout=20
                        )

                    if r.status_code == 200 and len(r.content) > 500:
                        text = self._parse_content(r.content, doc)
                        if text and len(text) > 200:
                            return {
                                "accession"  : acc,
                                "filing_text": self._extract_sections(
                                    text, filing_type
                                ),
                                "has_text"   : True
                            }
                except Exception:
                    pass

            # ── Method 2: Parse index.htm ───────────────────────────────
            try:
                time.sleep(0.15 + random.uniform(0, 0.1))
                index_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{acc}/{acc_fmt}-index.htm"
                )
                r2 = requests.get(
                    index_url, headers=self.HEADERS, timeout=20
                )

                if r2.status_code == 429:
                    time.sleep(60)
                    r2 = requests.get(
                        index_url, headers=self.HEADERS, timeout=20
                    )

                if r2.status_code != 200:
                    return empty

                # Parse index.htm to find primary document link
                soup     = BeautifulSoup(r2.content, "lxml")
                doc_href = None

                for tr in soup.find_all("tr"):
                    for td in tr.find_all("td"):
                        link = td.find("a", href=True)
                        if link:
                            href = link["href"]
                            if (
                                ("/Archives/" in href or
                                 href.endswith((".htm",".html"))) and
                                "index" not in href.lower() and
                                "ix?doc" not in href
                            ):
                                doc_href = href
                                break
                    if doc_href:
                        break

                # Fallback scan all links
                if not doc_href:
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        if (
                            "/Archives/edgar/data/" in href and
                            href.endswith((".htm",".html")) and
                            "index" not in href.lower()
                        ):
                            doc_href = href
                            break

                if not doc_href:
                    return empty

                # Build full URL
                if doc_href.startswith("/"):
                    full_url = f"https://www.sec.gov{doc_href}"
                elif doc_href.startswith("http"):
                    full_url = doc_href
                else:
                    full_url = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{cik_int}/{acc}/{doc_href}"
                    )

                # Fetch primary document
                time.sleep(0.15 + random.uniform(0, 0.1))
                r3 = requests.get(
                    full_url, headers=self.HEADERS, timeout=25
                )

                if r3.status_code == 429:
                    time.sleep(60)
                    r3 = requests.get(
                        full_url, headers=self.HEADERS, timeout=25
                    )

                if r3.status_code == 200 and len(r3.content) > 500:
                    text = self._parse_content(r3.content, full_url)
                    if text and len(text) > 200:
                        return {
                            "accession"  : acc,
                            "filing_text": self._extract_sections(
                                text, filing_type
                            ),
                            "has_text"   : True
                        }

            except Exception:
                pass

        return empty

    def _fetch_texts_threaded(self, filings: list) -> list:
        total = len(filings)
        print(f"\n  Phase 2: Fetching text for {total:,} filings "
              f"({self.N_TEXT_THREADS} threads)...")
        print(f"  Rate limit: ~{self.N_TEXT_THREADS * (1/0.15):.0f} req/sec "
              f"(SEC allows 10)")

        results = []
        start   = time.time()

        with ThreadPoolExecutor(
            max_workers=self.N_TEXT_THREADS
        ) as executor:
            futures = {
                executor.submit(self._fetch_text, f): i
                for i, f in enumerate(filings)
            }
            for i, future in enumerate(as_completed(futures)):
                try:
                    results.append(future.result())
                except Exception:
                    results.append({
                        "accession"  : filings[futures[future]]["accession"],
                        "filing_text": "",
                        "has_text"   : False
                    })

                if (i + 1) % 500 == 0:
                    elapsed  = (time.time() - start) / 60
                    pct      = 100 * (i + 1) / total
                    with_txt = sum(1 for r in results if r["has_text"])
                    rate     = (i + 1) / elapsed if elapsed > 0 else 1
                    eta      = (total - i - 1) / (rate * 60)
                    print(f"    {i+1:,}/{total:,} ({pct:.0f}%) | "
                          f"text: {with_txt:,} | "
                          f"{elapsed:.1f}min elapsed | "
                          f"ETA {eta:.0f}min")

        elapsed   = (time.time() - start) / 60
        with_text = sum(1 for r in results if r["has_text"])
        print(f"\n  ✓ Phase 2 complete in {elapsed:.1f} min")
        print(f"    Text fetched : {with_text:,}/{total:,} "
              f"({100*with_text/max(total,1):.1f}%)")
        return results

    # ------------------------------------------------------------------ #
    #  Main ingestion
    # ------------------------------------------------------------------ #
    def ingest(self) -> pd.DataFrame:
        start_total = time.time()

        # CIK map
        cik_map = self._build_cik_map()
        ticker_cik_pairs = [
            (t, cik_map[t.upper()])
            for t in self.tickers
            if t.upper() in cik_map
        ]
        print(f"  Matched : {len(ticker_cik_pairs)} tickers with CIK")
        print(f"  Skipped : {len(self.tickers) - len(ticker_cik_pairs)}")

        # Phase 1 — async index fetch
        print(f"\n  Phase 1: Async index fetch "
              f"({self.MAX_CONCURRENT} concurrent)...")
        p1_start    = time.time()
        loop        = asyncio.get_event_loop()
        all_filings = loop.run_until_complete(
            self._fetch_all_indexes_async(ticker_cik_pairs)
        )
        p1_time = (time.time() - p1_start) / 60
        print(f"  ✓ Phase 1 done in {p1_time:.1f} min")
        print(f"    Total filings found : {len(all_filings):,}")

        # Split
        text_filings = [
            f for f in all_filings if f["year"] >= self.TEXT_FROM_YEAR
        ]
        meta_filings = [
            f for f in all_filings if f["year"] < self.TEXT_FROM_YEAR
        ]
        print(f"    Text (2015+)        : {len(text_filings):,}")
        print(f"    Metadata only       : {len(meta_filings):,}")

        # Phase 2 — threaded text fetch
        text_results = self._fetch_texts_threaded(text_filings)
        text_map     = {r["accession"]: r for r in text_results}

        # Add empty text to metadata-only filings
        for f in meta_filings:
            f["filing_text"] = ""
            f["has_text"]    = False

        # Build final rows
        rows = []
        for f in all_filings:
            acc      = f["accession"]
            txt      = text_map.get(acc, {})
            rows.append({
                "ticker"      : f["ticker"],
                "cik"         : f["cik"],
                "filing_type" : f["filing_type"],
                "filing_date" : f["filing_date"],
                "year"        : f["year"],
                "month"       : f["month"],
                "accession_no": acc,
                "has_text"    : txt.get("has_text", False),
                "text_length" : len(txt.get("filing_text", "")),
                "filing_text" : txt.get("filing_text", ""),
                "ingested_at" : datetime.now(timezone.utc).isoformat()
            })

        total_time = (time.time() - start_total) / 60
        print(f"\n{'='*50}")
        print(f"Ingestion complete in {total_time:.1f} minutes")
        print(f"  Total records : {len(rows):,}")
        print(f"  With text     : {sum(1 for r in rows if r['has_text']):,}")
        print(f"  Metadata only : {sum(1 for r in rows if not r['has_text']):,}")
        print(f"  Failed        : {len(self.failed)}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Spark + Delta
    # ------------------------------------------------------------------ #
    def _to_spark(self, pdf: pd.DataFrame):
        schema = StructType([
            StructField("ticker",       StringType(),  False),
            StructField("cik",          StringType(),  True),
            StructField("filing_type",  StringType(),  False),
            StructField("filing_date",  StringType(),  True),
            StructField("year",         IntegerType(), False),
            StructField("month",        IntegerType(), False),
            StructField("accession_no", StringType(),  True),
            StructField("has_text",     BooleanType(), True),
            StructField("text_length",  IntegerType(), True),
            StructField("filing_text",  StringType(),  True),
            StructField("ingested_at",  StringType(),  True),
        ])
        return self.spark.createDataFrame(pdf, schema=schema)

    def write_delta(self, sdf) -> None:
        print(f"\nWriting to Delta: {self.path}")
        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("filing_type", "year")
            .save(self.path)
        )
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        print("Write complete ✓")

    def optimize(self) -> None:
        print("\nOPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled", "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.path}` RETAIN 168 HOURS"
        )
        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.path}`"
        ).select("numFiles", "sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    def validate(self) -> None:
        print("\n" + "="*50)
        print("VALIDATION — Bronze SEC Filings")
        print("="*50)
        df    = self.spark.read.format("delta").load(self.path)
        total = df.count()
        print(f"  Total rows     : {total:,}")
        print(f"  Unique tickers : {df.select('ticker').distinct().count()}")
        df.groupBy("filing_type").agg(
            F.count("*").alias("total"),
            F.sum(F.col("has_text").cast("int")).alias("with_text"),
            F.avg("text_length").cast("int").alias("avg_chars")
        ).orderBy("filing_type").show()
        df.filter(F.col("year") >= 2015) \
          .groupBy("year").count().orderBy("year").show(15)
        print("Validation PASSED ✓")

# COMMAND ----------

# Load tickers
ohlcv_df      = spark.read.format("delta").load(
    f"{BASE_PATH}/bronze/delta/ohlcv"
)
sp500_tickers = [
    r.ticker for r in
    ohlcv_df.select("ticker").distinct().collect()
]
print(f"Tickers loaded: {len(sp500_tickers)}")

# Create ingestion object
ingestion = BronzeSECFilingsFinal(
    spark           = spark,
    base_path       = BASE_PATH,
    tickers         = sp500_tickers,
    checkpoint_path = CHECKPOINT_PATH
)

# Run ingestion
pdf = ingestion.ingest()

# ── Save checkpoint IMMEDIATELY ────────────────────────────────────────
pdf.to_parquet(CHECKPOINT_PATH, index=False)
print(f"\nCheckpoint saved ✓")
print(f"  Path : {CHECKPOINT_PATH}")
print(f"  Size : {os.path.getsize(CHECKPOINT_PATH)/1e6:.1f} MB")
print(f"  Rows : {len(pdf):,}")

# COMMAND ----------

print(f"Records in memory: {len(pdf):,}")
print(f"With text: {pdf['has_text'].sum():,}")

# Convert to Spark
sdf = ingestion._to_spark(pdf)

# Write to Delta
ingestion.write_delta(sdf)

# Optimize
ingestion.optimize()

# Validate
ingestion.validate()

print("\nBronze SEC Filings COMPLETE ✓")

# COMMAND ----------

sdf = ingestion._to_spark(pdf)
ingestion.write_delta(sdf)
ingestion.optimize()
ingestion.validate()
print("\nBronze SEC Filings COMPLETE ✓")