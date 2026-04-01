# Databricks notebook source
# MAGIC %pip install yfinance --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, date
import pandas as pd
import yfinance as yf
import time
import os

# COMMAND ----------

STORAGE_ACCOUNT  = "multisignalalphaeng"
CONTAINER        = "quant-lakehouse"
ADLS_KEY         = dbutils.secrets.get(scope="quant-scope", key="adls-key-01")
CHECKPOINT_PATH  = "/tmp/russell2000_checkpoint.parquet"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    ADLS_KEY
)

BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
print(f"Config loaded ✓")
print(f"Checkpoint : {CHECKPOINT_PATH}")

# COMMAND ----------

class BronzeRussell2000Ingestion:
    """
    Bronze Russell 2000 daily OHLCV ingestion.
    - ~1800 active small cap tickers
    - Daily bars from 1993 to present
    - Checkpoint saved immediately after fetch
    - Handles delisted tickers gracefully
    - Fixes duplicate MultiIndex columns
    Expected size : ~400-500MB Delta compressed
    Expected time : ~25-35 minutes
    """

    RUSSELL2000_TICKERS = [
        "AADI","AAIN","AAON","AAPB","AAPD","AATC","ABAT","ABCB","ABCL",
        "ABEO","ABGI","ABIO","ABLD","ABOS","ABPX","ABR","ABSI","ABST",
        "ABTX","ABVC","ABVX","ACAB","ACAD","ACAX","ACBA","ACCD","ACET",
        "ACEV","ACGN","ACHC","ACHR","ACHV","ACIU","ACKF","ACLX","ACMR",
        "ACNB","ACNT","ACOR","ACRS","ACRV","ACRX","ACST","ACTD","ACTG",
        "ACTU","ACVA","ACVF","ACVI","ACXP","ADAG","ADAP","ADCT","ADEA",
        "ADEX","ADGM","ADIL","ADMA","ADMP","ADMS","ADNT","ADOC","ADPT",
        "ADSE","ADTX","ADUS","ADV","ADVM","ADVS","ADXN","ADXS","AEAC",
        "AEAE","AEHR","AEIS","AEMD","AENZ","AERI","AESE","AEYE","AFAR",
        "AFBI","AFCG","AFIB","AFMD","AFRI","AFRY","AFTR","AFYA","AGAE",
        "AGBA","AGEN","AGFS","AGFY","AGIL","AGIO","AGLY","AGMH","AGOX",
        "AGPX","AGRI","AGRO","AGRX","AGSD","AGTC","AGTI","AGYS","AHCO",
        "AHGP","AHHI","AHPA","AHPI","AHRN","AHRX","AIFU","AIKI","AIMD",
        "AINC","AINV","AIRC","AIRG","AIRI","AIRJ","AIRT","AIXI","AJRD",
        "AKBA","AKCA","AKER","AKLI","AKRO","AKTS","AKTX","AKUS","AKYA",
        "ALAC","ALBT","ALCO","ALDX","ALEC","ALGT","ALIM","ALJJ","ALKS",
        "ALKT","ALLK","ALLO","ALLR","ALLT","ALNY","ALOT","ALPA","ALPN",
        "ALRM","ALRS","ALSA","ALSE","ALSK","ALSN","ALTA","ALTG","ALTO",
        "ALTR","ALTU","ALUR","ALUS","ALVR","ALVO","ALXO","ALYA","ALZN",
        "AMAL","AMAM","AMAO","AMBA","AMBC","AMBO","AMBP","AMCX","AMDI",
        "AMED","AMEH","AMER","AMHC","AMIX","AMKR","AMMO","AMNB","AMNT",
        "AMOR","AMOT","AMPE","AMPH","AMPI","AMPL","AMPO","AMRB","AMRC",
        "AMRK","AMRN","AMRX","AMSC","AMSF","AMSG","AMST","AMSWA","AMTB",
        "AMTD","AMTI","AMTX","AMWD","AMWL","AMXT","ANAB","ANDE","ANEB",
        "ANGI","ANGL","ANGN","ANGT","ANIK","ANIP","ANIX","ANNX","ANPC",
        "ANTE","ANTX","ANVS","ANY","AOGN","AORT","APAM","APCA","APDN",
        "APEI","APEN","APGE","APGN","APGT","APLD","APLE","APLS","APLT",
        "APMA","APMI","APOG","APOP","APPF","APPH","APPI","APRE","APRT",
        "APTO","APTS","APTX","APVO","APWC","APXI","APYX","AQMS","AQST",
        "AQTB","ARAV","ARBB","ARBE","ARBK","ARBN","ARCE","ARCO","ARCT",
        "ARDS","AREC","ARGX","ARHS","ARID","ARIS","ARIZ","ARKO","ARKR",
        "ARLO","ARLP","ARMP","ARNC","AROA","AROW","ARQT","ARQQ","ARRW",
        "ARRY","ARTE","ARTL","ARTNA","ARTW","ARVL","ARVN","ARWR","ARYD",
        "ARYE","ARZN","ASAI","ASAL","ASAR","ASAX","ASBA","ASBI","ASCA",
        "ASCK","ASEP","ASET","ASFI","ASFX","ASGA","ASGI","ASIX","ASKE",
        "ASLN","ASMB","ASMD","ASND","ASNS","ASOB","ASPC","ASPS","ASPU",
        "ASRT","ASRV","ASTC","ASTE","ASTG","ASTI","ASTL","ASTR","ASTS",
        "ASUR","ASYS","ATAI","ATAQ","ATAR","ATAX","ATCX","ATEC","ATEN",
        "ATEX","ATGN","ATHA","ATHE","ATHL","ATIF","ATIP","ATIS","ATIX",
        "ATKG","ATKR","ATLC","ATLO","ATMC","ATMD","ATMP","ATMU","ATNI",
        "ATNM","ATNT","ATOM","ATOS","ATPC","ATPL","ATPR","ATRC","ATRI",
        "ATRM","ATRS","ATSG","ATSI","ATST","ATUL","ATUS","ATVC","ATXG",
        "ATXI","ATXS","ATYR","AUBN","AUDC","AUGX","AUID","AUPH","AUTL",
        "AUUD","AUVI","AVAH","AVBH","AVCO","AVDL","AVDX","AVEC","AVGR",
        "AVHI","AVID","AVIG","AVIN","AVIR","AVIV","AVNW","AVPT","AVRO",
        "AVTA","AVTE","AVTS","AVXL","AVXT","AWAY","AXDX","AXGN","AXIL",
        "AXNX","AXSM","AXTA","AXTI","AYRO","AYTU","AZEK","AZEN","AZPN",
        "AZTA","AZUL","AZYO","BACK","BAER","BAFN","BAND","BANF","BANL",
        "BANR","BANT","BANX","BARK","BARN","BASE","BATL","BAYA","BBAI",
        "BBCP","BBDC","BBGI","BBIO","BBLN","BBSI","BBTO","BBUC","BBWI",
        "BCAB","BCAL","BCBP","BCDA","BCEL","BCLI","BCML","BCNB","BCOW",
        "BCPC","BCSA","BCSG","BCSF","BCUS","BCYC","BDAY","BDGE","BDMD",
        "BDPT","BDSX","BDTX","BEAT","BECN","BEEM","BEIN","BELFA","BELFB",
        "BELR","BENF","BFAC","BFAM","BFIN","BFLY","BFRI","BFST","BGFV",
        "BGNE","BGSF","BGXX","BHAC","BHAT","BHAV","BHIL","BHLB","BHRB",
        "BHSE","BHVN","BIAF","BIBE","BICO","BIGC","BIGG","BILI","BIMI",
        "BIOA","BIOC","BIOF","BIOG","BIOL","BIOR","BIOS","BIOX","BIRD",
        "BIRI","BKFC","BKFG","BKKT","BKSY","BKTI","BKVE","BKYI","BLBX",
        "BLCM","BLCO","BLDP","BLDR","BLEU","BLEW","BLFS","BLIN","BLKB",
        "BLMN","BLND","BLNK","BLPH","BLRX","BLTE","BLTS","BLUE","BLUA",
        "BLUD","BLUF","BLUM","BLUR","BMAC","BMAP","BMEA","BMEZ","BMRA",
        "BMTC","BNAI","BNED","BNET","BNGO","BNIX","BNKL","BNOX","BNRE",
        "BNRG","BNSO","BNTC","BNXG","BNYF","BOCN","BODI","BOKF","BOLT",
        "BOMN","BOOM","BOOT","BORR","BOSC","BOTJ","BOWN","BPMC","BPOP",
        "BPRN","BPTH","BPTS","BRAC","BRAG","BRAM","BRBR","BRBS","BRCN",
        "BRDS","BREA","BREZ","BRFH","BRFS","BRID","BRKL","BRKR","BRLT",
        "BRMK","BROG","BRTX","BRVS","BRWC","BSAC","BSAQ","BSBK","BSBR",
        "BSET","BSFC","BSGM","BSIG","BSIN","BSKR","BSRR","BSVN","BTAI",
        "BTBT","BTCS","BTCY","BTDG","BTEL","BTMD","BTON","BTSG","BTTX",
        "BTUS","BURU","BUSE","BVFL","BWAY","BWEN","BWFG","BWIN","BWMN",
        "BWXT","BYFC","BYND","BYRN","BYSI","BZFD","BZUN","CAAS","CABA",
        "CABO","CACC","CACO","CACT","CADC","CADL","CAFA","CAKE","CALA",
        "CALC","CALM","CALX","CAMP","CAMT","CANE","CANO","CANF","CANG",
        "CANI","CANK","CANN","CANT","CAPR","CARE","CARG","CARM","CARO",
        "CARS","CARV","CASA","CASH","CASI","CASM","CASS","CAST","CATC",
        "CATO","CATX","CBAN","CBAT","CBFV","CBLI","CBMG","CBNK","CBPO",
        "CBRL","CBSH","CBTX","CBYL","CCAP","CCAX","CCBG","CCCC","CCCS",
        "CCEP","CCHE","CCIX","CCLD","CCLP","CCNE","CCNX","CCOC","CCOI",
        "CCRD","CCRN","CCSI","CCTS","CCVI","CDAQ","CDLX","CDMO","CDNA",
        "CDRE","CDRO","CDTX","CDXC","CDXS","CDZI","CEAD","CECO","CEER",
        "CEIN","CELC","CELH","CELL","CELU","CELZ","CENT","CENTA","CENX",
        "CEPU","CERS","CERT","CESC","CETU","CEVA","CFAC","CFBK","CFFI",
        "CFFN","CFFS","CFLT","CFMS","CFNB","CFSB","CGEM","CGEN","CGNT",
        "CGOB","CGON","CGRO","CGTX","CHCO","CHDN","CHEA","CHEF","CHEK",
        "CHGG","CHMG","CHMI","CHNR","CHPT","CHRD","CHRS","CHRY","CHSN",
        "CHUY","CIFR","CIGI","CINC","CING","CINT","CISO","CIVB","CLBR",
        "CLBT","CLCO","CLDT","CLDX","CLEU","CLFD","CLGN","CLIR","CLNN",
        "CLNV","CLOE","CLPS","CLPT","CLRB","CLRC","CLRO","CLSK","CLST",
        "CLVR","CLVS","CLWT","CMAX","CMBT","CMCO","CMCT","CMDX","CMLS",
        "CMMB","CMND","CMNF","CMPO","CMPR","CMPS","CMRA","CMRX","CMSA",
        "CMTG","CNDB","CNET","CNIC","CNMD","CNNB","CNOB","CNSL","CNSP",
        "CNTA","CNTB","CNTQ","CNTY","CNVS","CNXA","CNXC","CNXN","COCP",
        "CODA","CODX","COFS","COGT","COIN","COKE","COMS","CONN","COOL",
        "CORT","COSM","COVA","COWN","CPAA","CPAR","CPBI","CPIX","CPOP",
        "CPRX","CPSH","CPSI","CPSS","CPTK","CPUH","CRAI","CRBP","CRCT",
        "CRCW","CRDF","CRDL","CRDO","CREG","CRGE","CRGX","CRGY","CRMD",
        "CRMT","CRNC","CRNX","CRON","CROX","CRSP","CRSR","CRTD","CRTO",
        "CRTX","CRUS","CRVL","CRVS","CRWS","CSBR","CSBS","CSCW","CSGS",
        "CSII","CSIQ","CSLM","CSLR","CSOD","CSSE","CSTA","CSTE","CSTL",
        "CSTR","CSTX","CTBI","CTGO","CTHR","CTIB","CTIC","CTLP","CTMX",
        "CTNM","CTNT","CTOS","CTRE","CTRM","CTRN","CTSO","CUBA","CUBB",
        "CUBE","CUBI","CUBT","CUEN","CULL","CULP","CURO","CURV","CUTR",
        "CUVA","CVBF","CVCO","CVCY","CVEO","CVGI","CVGW","CVII","CVKD",
        "CVLG","CVLY","CVNX","CVOP","CVRX","CVTI","CWBC","CWBR","CWCO",
        "CWEN","CWST","CXAI","CXDO","CYCC","CYCN","CYRN","CYRX","CYTH",
        "CYTK","CYTO","CZFS","DADA","DAIO","DAKT","DALI","DALT","DARE",
        "DATS","DBGI","DBIO","DBRG","DBVT","DCBO","DCFC","DCGO","DCOM",
        "DCPH","DCRD","DCRX","DCTH","DDMX","DDOG","DECA","DECK","DENN",
        "DERR","DFFN","DFIN","DFLI","DGHI","DGIO","DGLY","DHBC","DHIL",
        "DHTE","DIBS","DIFI","DIGS","DIOD","DJCO","DKNG","DLHC","DLNG",
        "DLPN","DLTH","DMRC","DMTK","DNLI","DNMR","DNOW","DNUT","DOCU",
        "DOMO","DORM","DOUG","DPSI","DRCT","DRIO","DRMA","DRRX","DRVN",
        "DSGN","DSGR","DSKE","DSNY","DSSI","DTIL","DTRT","DUOL","DVAX",
        "DVCR","DWSN","DXLG","DXPE","DXYN","DYAI","DYNT","DYSL","DZSI",
        "EACO","EARN","EAST","EATZ","EBIX","EBMT","EBTC","ECBK","ECDA",
        "ECMB","ECOR","ECPG","ECVT","EDAP","EDBL","EDIT","EDNT","EDRY",
        "EDSA","EDUC","EEFT","EFSC","EFSH","EGAN","EGBN","EGHT","EGIO",
        "EGLX","EGOX","EGRX","EHAB","EHTH","EINC","ELAB","ELEV","ELIF",
        "ELIQ","ELMD","ELME","ELOX","EMBC","EMCG","EMKR","EMLD","EMNT",
        "EMOW","EMPR","EMXC","ENER","ENOV","ENSC","ENSG","ENTA","ENTX",
        "ENVB","ENVS","ENVX","EOLS","EOSE","EPIX","EPOW","EPRT","EQBK",
        "EQRX","ERAS","ERES","ERGO","ERII","ERNA","EROS","ESAB","ESBA",
        "ESCA","ESEA","ESGL","ESLT","ESMT","ESOA","ESPR","ESRT","ESSA",
        "ESTE","ESXB","ETAO","ETCC","ETNB","ETON","ETSY","EVAX","EVBG",
        "EVBN","EVCO","EVEI","EVER","EVEX","EVGN","EVGO","EVGR","EVGX",
        "EVIO","EVLV","EVMO","EVOP","EVRI","EVTL","EVVL","EWBC","EWCZ",
        "EWTX","EXAI","EXEL","EXFY","EXLS","EXPI","EXPO","EXTO","EXTR",
        "EYEG","EYEN","EYES","EYPT","EZFL","EZGO","EZPW","FACA","FACT",
        "FANH","FARM","FARO","FAST","FATE","FATH","FBIO","FBIZ","FBRX",
        "FCAP","FCBC","FCCO","FCEL","FCFS","FCRD","FCRX","FCUV","FDBC",
        "FDMT","FDUS","FEAM","FEBO","FEDU","FEIM","FELE","FEMY","FENF",
        "FENV","FERN","FFBC","FFWM","FGBI","FGEN","FGMC","FHLT","FHTX",
        "FIGS","FISI","FISR","FIVE","FIVN","FIXX","FIZZ","FKWL","FLCX",
        "FLGC","FLIC","FLME","FLNC","FLNT","FLWR","FLXS","FLYE","FMBI",
        "FMBH","FMCB","FMST","FMTX","FNCH","FNCX","FNKO","FNLC","FNWB",
        "FNWD","FOCS","FOLD","FONR","FORD","FORL","FORM","FORR","FOSL",
        "FOXF","FPAY","FRAF","FRBA","FRBI","FRBK","FRBN","FRCB","FRGE",
        "FRGT","FRLA","FRME","FRMO","FRMS","FROG","FRPH","FRPT","FRSG",
        "FRST","FRSX","FRTX","FRZA","FSBC","FSBW","FSCO","FSFG","FSLY",
        "FSMB","FSNB","FSTR","FTCI","FTDR","FTEK","FTFT","FTHM","FTRE",
        "FTSI","FTUR","FUBO","FUFU","FULC","FULT","FUNC","FUND","FURY",
        "FUSB","FUSN","FUTU","FWBI","FWONA","FWONK","FWRG","FXNC","FYBR",
        "GABC","GADS","GAIA","GAIN","GALT","GAMB","GAME","GATO","GBCI",
        "GBIO","GBLI","GBNY","GCBC","GCMG","GCNL","GCTS","GDNP","GDOT",
        "GDYN","GENC","GENI","GENK","GENY","GEOS","GEVI","GFAI","GFED",
        "GGAL","GHLD","GHRS","GHSI","GIAX","GILT","GLAD","GLBE","GLBS",
        "GLDD","GLHA","GLLI","GLMD","GLNG","GLOB","GLPG","GLRE","GLSI",
        "GLTO","GLUE","GMRE","GNFT","GNLX","GNPX","GNSS","GNTX","GNTY",
        "GNUS","GOEV","GOGO","GOOD","GORV","GPCO","GPCR","GPMT","GPRE",
        "GPRK","GPRO","GRAB","GRBK","GRCE","GREE","GRFS","GRIN","GRND",
        "GROM","GROV","GROW","GRPN","GRTS","GRTX","GRWG","GRWN","GSBC",
        "GSIT","GSMG","GTBP","GTEC","GTHR","GTIM","GTLB","GTLS","GTRX",
        "GWAV","GWRS","HAFC","HALN","HALO","HARP","HARR","HASI","HAYN",
        "HBCP","HBIO","HBMD","HCAT","HCCI","HCKT","HCNB","HCSG","HCTI",
        "HCVX","HCWB","HDSN","HEPS","HERC","HFFG","HFWA","HGBL","HGEN",
        "HGLB","HGTY","HIBB","HIFS","HIHO","HILS","HIMX","HIMS","HIPO",
        "HKIT","HLAH","HLAN","HLBZ","HLCO","HLGN","HLIO","HLLY","HLMN",
        "HLNE","HLTH","HLVX","HMCO","HMCX","HMPT","HMST","HMTV","HNNA",
        "HNRG","HOFV","HOFT","HOLI","HOMB","HOME","HONE","HOOK","HOTH",
        "HOUS","HPCO","HPNN","HPRO","HRMY","HROW","HRYU","HSCS","HSDT",
        "HSHP","HSII","HSKA","HSON","HSPG","HSTO","HTBK","HTBI","HTBX",
        "HTCR","HTGC","HTGM","HTHT","HTIA","HTLD","HTLF","HTOO","HTRM",
        "HUBG","HUDI","HUIZ","HURC","HURN","HUSN","HWBK","HWCC","HWKN",
        "HYLN","HYMC","HYPR","IAGG","IART","IBCP","IBEX","IBIO","IBKR",
        "IBOC","ICAD","ICBK","ICCC","ICCH","ICCM","ICCT","ICFI","ICHR",
        "ICMB","ICPT","ICUI","IDBA","IDCC","IDEX","IDRA","IDYA","IFIN",
        "IFRX","IGIC","IGMS","IGPK","IHRT","IINN","IKNA","IKNX","ILPT",
        "IMAB","IMAX","IMBI","IMCC","IMCR","IMGN","IMGO","IMKTA","IMMP",
        "IMNN","IMRN","IMRX","IMTE","IMTX","IMUX","IMVT","IMXI","INAB",
        "INBK","INBS","INCR","INDB","INDI","INDT","INEO","INFN","INFU",
        "INGN","INHD","INKT","INLX","INMB","INMD","INNO","INNV","INOD",
        "INPX","INSE","INSG","INSM","INSP","INSW","INTA","INTG","INTT",
        "INTZ","INVA","INVE","INVX","INZY","IONQ","IONR","IONS","IOVA",
        "IPDN","IPGP","IPHA","IPIX","IPSC","IPSI","IQMD","IRBT","IRCP",
        "IREN","IRET","IRMD","IRNT","IRON","IROQ","IRTC","IRUS","IRWD",
        "ISAB","ISEE","ISLE","ISNS","ISPC","ISPR","ISSC","ISTR","ISUN",
        "ITCI","ITGR","ITIC","ITRM","ITRN","ITRI","ITUB","IVAC","IVAN",
        "IVDA","IVDN","JBGS","JBLU","JBTX","JCSE","JFIN","JFNB","JILL",
        "JJSF","JOBY","JODN","JOUT","JRNC","JRSH","JRVR","JSPR","JYNT",
        "KALA","KALI","KALU","KALV","KARO","KBAL","KBNT","KBSF","KDLY",
        "KDMN","KDNY","KEFI","KELYA","KELYB","KERN","KFRC","KIDS","KIND",
        "KINS","KIRK","KLDI","KLTR","KMDA","KMPH","KNDI","KNDL","KNSA",
        "KNSL","KNTE","KODK","KORE","KOSS","KPLT","KPTI","KRMD","KRON",
        "KRNT","KROS","KRTX","KRUS","KRYS","KSCP","KTCC","KTOS","KTRA",
        "KVHI","KWAC","KYMR","LAKE","LAND","LASR","LATG","LAUR","LAZR",
        "LBAI","LBPH","LBTYA","LBTYB","LBTYK","LCII","LCNB","LCUT","LDHA",
        "LEGH","LESL","LEXX","LFAC","LFCR","LFST","LFUS","LFVN","LGFA",
        "LGFB","LGHL","LGND","LGVN","LHCG","LHDX","LIDR","LIEN","LINC",
        "LINK","LIQT","LITB","LITE","LITM","LIVX","LLAP","LLNW","LMAT",
        "LMFA","LMND","LMST","LNFA","LNKB","LNTH","LOAN","LOCO","LODE",
        "LOGI","LOOP","LOPE","LOVE","LPCN","LPLA","LPRO","LPSN","LPTH",
        "LPTX","LQDA","LQDT","LRHC","LRMR","LRTX","LSBK","LSCC","LSEA",
        "LSPR","LSTR","LTBR","LTHM","LTRY","LTRX","LUMN","LUMO","LUNA",
        "LWAY","LYEL","LYRA","LYTS","MACA","MACS","MADD","MAPS","MARA",
        "MARK","MASA","MASC","MASI","MATV","MAYS","MBCN","MBII","MBIN",
        "MBIO","MBLY","MBOT","MBUU","MBWM","MCBC","MCEM","MCFT","MCHX",
        "MCLD","MCRB","MCRI","MCRX","MCSF","MDAI","MDGL","MDGS","MDIG",
        "MDNA","MDP","MDSP","MDVL","MDXG","MEAN","MEDP","MEDS","MEGL",
        "MEOH","MERC","MFAC","MFIN","MFNC","MGEE","MGEN","MGLD","MGNX",
        "MGPI","MGRC","MGRM","MGTA","MGTI","MGYR","MHLA","MHLD","MHUA",
        "MICS","MIDD","MIGI","MIKP","MIMO","MINM","MIRA","MIRM","MIST",
        "MITQ","MITO","MITT","MIXT","MKFG","MKSI","MLCO","MLFB","MLGO",
        "MLNK","MLSS","MMAC","MMEX","MMND","MMSI","MNCL","MNDO","MNKD",
        "MNOV","MNPR","MNRO","MNTK","MNTN","MNTX","MODD","MODG","MODI",
        "MODV","MOFG","MOGO","MOLN","MOMO","MOND","MONG","MONN","MORF",
        "MORN","MOSY","MOTS","MPAA","MPLN","MPTI","MRAM","MRBK","MRCC",
        "MRCY","MRIN","MRNJ","MRNS","MRSN","MRTN","MRTX","MRUS","MSBI",
        "MSEX","MSGE","MSGM","MSRT","MSTO","MSTR","MSVB","MTAL","MTCN",
        "MTCP","MTCR","MTEX","MTHQ","MTNB","MTNS","MTOR","MTRX","MTSI",
        "MTTR","MULN","MURA","MVBF","MVLA","MVOR","MVST","MWRK","MXCT",
        "MYFW","MYGN","MYMD","MYMS","MYND","MYOS","MYRG","MYTE","NABL",
        "NACP","NAFS","NAII","NAMI","NAOV","NAPA","NARI","NATH","NBTB",
        "NCDL","NCNA","NCPL","NCRA","NCSM","NCTY","NDLS","NDRA","NDSN",
        "NDVG","NEGG","NEOG","NEON","NEOS","NEOT","NEPH","NERD","NETI",
        "NETX","NEWR","NEWT","NEXT","NFBK","NFGC","NFLD","NGNE","NGVC",
        "NGVT","NHWK","NICK","NIMD","NINE","NIOB","NITO","NKGN","NKLA",
        "NKTR","NLOP","NLSP","NMCO","NMFC","NMIH","NMKI","NMRA","NMRK",
        "NMTR","NNBR","NNDM","NNOX","NODK","NOMD","NONS","NORA","NOVA",
        "NOVN","NOVS","NPKI","NPWR","NRDS","NRDY","NREF","NRIM","NRIX",
        "NRXP","NSEC","NSLM","NSPR","NSSC","NSTG","NSYS","NTBL","NTCT",
        "NTGR","NTIC","NTIP","NTLA","NTNX","NTST","NTUS","NUSC","NUTX",
        "NUVA","NUZE","NVET","NVAX","NVCN","NVCR","NVCT","NVEC","NVEI",
        "NVFY","NVGS","NVNI","NVNO","NVOS","NVRI","NVRO","NVST","NWBI",
        "NWFL","NWGL","NWLI","NWPX","NWTN","NXGL","NXPL","NXRT","NXST",
        "NXTC","NXTP","NYMT","NYMX","NYXH","OABI","OBNK","OBSV","OCFC",
        "OCGN","OCIO","OCLR","OCSL","OCTO","OCUL","OCUP","ODMO","OFED",
        "OFFH","OFIX","OFLX","OIIM","OKLO","OKTA","OLMA","OLPX","OMAB",
        "OMCL","OMEG","OMER","OMEX","OMGA","OMID","OMNV","ONCS","ONCT",
        "ONCX","ONDS","ONEM","ONEW","ONFO","ONMD","ONON","ONTF","OOMA",
        "OPBK","OPEN","OPFI","OPHC","OPKO","OPRA","OPRX","OPTN","OPTX",
        "ORBC","ORGO","ORGS","ORMP","ORRF","ORSN","ORTX","OSBC","OSCR",
        "OSEA","OSIS","OSPA","OSTK","OSUR","OTEL","OTLK","OTMO","OTTR",
        "OVBC","OVID","OVLY","OWLT","OXBR","OXSQ","OYST","PACI","PACK",
        "PAGS","PAHC","PALI","PALT","PANL","PASG","PATI","PATK","PAYS",
        "PBAX","PBCP","PBFS","PBHC","PBIP","PBPB","PBTS","PBTX","PCBC",
        "PCBK","PCCO","PCSA","PCTI","PCVX","PDCE","PDCO","PDEX","PDFS",
        "PDLB","PDLN","PDSB","PEAR","PECK","PEGY","PESI","PETQ","PETZ",
        "PFBC","PFBI","PFHD","PFIE","PFIS","PFLT","PFMT","PFNX","PGEN",
        "PGNY","PHAT","PHGE","PHIO","PHMB","PHMD","PHUN","PHVS","PLBC",
        "PLBY","PLCE","PLAY","PLMR","PLNT","PLOW","PLPC","PLRX","PLTK",
        "PLTO","PLUG","PLUR","PLUS","PLXS","PNFP","PNNT","PNTM","PNVL",
        "POET","POLA","POLY","PODD","PORT","POWI","POWL","PPBI","PPBT",
        "PPIH","PPSI","PRAA","PRAX","PRCH","PRCT","PRDO","PRFT","PRFX",
        "PRGN","PRGO","PRGS","PRLB","PRME","PROS","PROV","PRPH","PRPL",
        "PRPO","PRQR","PRSE","PRST","PRTG","PRTH","PRTK","PRTS","PRTY",
        "PRVA","PRVB","PSEC","PSEN","PSHG","PSMT","PSNY","PSQH","PSTG",
        "PSTV","PTBK","PTCT","PTEN","PTGX","PTIX","PTLO","PTNR","PTPI",
        "PTSG","PTVE","PVBC","PVRA","PWFL","PWOD","PWSC","PXLW","PXMD",
        "PYXS","QBTS","QCRH","QDEL","QETA","QFIN","QGEN","QMCO","QNCX",
        "QNRX","QRHC","QRTEA","QRTEB","QRVO","QTTB","QUBT","QUIK","QURE",
        "RADA","RADI","RAIL","RAND","RAPT","RARE","RAVE","RBBN","RBCA",
        "RBCAA","RCAT","RCFA","RCKT","RCKY","RCMT","RCOR","RCRT","RCUS",
        "RDCM","RDHL","RDFN","RDIB","RDUS","RDVT","RDWR","REAL","REAX",
        "REBN","REDU","REED","REFI","REFR","REGH","REKR","RELY","RENB",
        "RENE","RENT","REPL","REPX","RERE","RETA","RETO","REVE","REVG",
        "REZI","RFIL","RGCO","RGEN","RGLD","RGLS","RGNX","RGRX","RGTI",
        "RHBK","RICK","RIGL","RILY","RIOT","RIVN","RKDA","RKLB","RLAY",
        "RLGT","RLMD","RMBI","RMCF","RMCO","RMED","RMNI","RNLX","RNST",
        "ROAD","ROBS","ROCP","ROCK","ROIC","ROLL","RONS","ROSC","ROSE",
        "ROVR","RPAY","RPHM","RPID","RPTX","RRGB","RRHI","RSEM","RSLS",
        "RSSS","RTCO","RTGN","RTIC","RTRN","RTRX","RUBY","RUSHA","RUSHB",
        "RUTH","RVSB","RWLK","RXMD","RXRA","RXRX","RXST","RYAM","RYTM",
        "RZLT","SACH","SAFE","SAGA","SAGE","SAIA","SALF","SALM","SAMG",
        "SANA","SANG","SANN","SANM","SASI","SASR","SATL","SATS","SAVA",
        "SAVN","SBAR","SBAT","SBBP","SBCF","SBEV","SBFG","SBGI","SBLK",
        "SBOW","SBRA","SBSI","SBSW","SCHL","SCHN","SCKT","SCLE","SCLX",
        "SCNX","SCPH","SCPL","SCPS","SCSC","SCVL","SCWO","SDGR","SDIG",
        "SDOT","SEAC","SEAL","SEEL","SEER","SEHI","SELB","SEMA","SEMR",
        "SEND","SENEA","SENEB","SENS","SERA","SERV","SFBS","SFET","SFIO",
        "SFNC","SFST","SGBX","SGHC","SGLY","SGMO","SGMT","SGRP","SGRY",
        "SGTX","SHBI","SHCA","SHCR","SHFS","SHLS","SHMD","SHMP","SHOT",
        "SHPW","SHSP","SIBN","SIDU","SIGA","SIGI","SIMO","SINT","SIOX",
        "SIRC","SIRE","SISI","SITM","SIVO","SKGR","SKIN","SKIS","SKWD",
        "SKYE","SKYT","SKYW","SLCA","SLDB","SLDP","SLGG","SLGL","SLGN",
        "SLHG","SLIM","SLMN","SLND","SLNG","SLNO","SLQT","SLRC","SLRN",
        "SLRX","SMBC","SMCO","SMFL","SMID","SMIH","SMLP","SMMT","SMMF",
        "SMPL","SMSI","SMTK","SMTX","SNAX","SNBR","SNCR","SNCY","SNDA",
        "SNDL","SNDR","SNEX","SNFCA","SNMP","SNOA","SNPX","SNSE","SNSR",
        "SNVR","SOBR","SOFI","SOHO","SOHU","SOLO","SONX","SOPM","SOPA",
        "SOTK","SOWG","SPCE","SPFI","SPHS","SPLP","SPNT","SPOK","SPPL",
        "SPRA","SPRC","SPRO","SPRT","SPRX","SPRY","SPSC","SPWH","SQFT",
        "SQNS","SQSP","SRAC","SRAX","SRCE","SRGA","SRPT","SRRK","SRTS",
        "SSBI","SSEX","SSFN","SSIC","SSII","SSNC","SSNT","SSRM","SSTI",
        "SSYS","STAA","STAF","STAG","STBA","STBX","STCN","STEM","STEP",
        "STGW","STIM","STJM","STKS","STLV","STNE","STNG","STRE","STRM",
        "STRN","STRR","STRS","STRT","STSS","STVN","STWD","STXS","SUMO",
        "SUNL","SUPN","SURF","SURG","SVNA","SVRA","SWAG","SWBI","SWKH",
        "SXTP","SYBT","SYBX","SYNA","SYNH","SYNX","SYPR","SYRA","SYRE",
        "SYRS","SYTA","TACT","TALN","TALO","TANH","TAOP","TARA","TARO",
        "TARS","TAST","TBNK","TBPH","TBTX","TCBC","TCBK","TCBX","TCFC",
        "TCMD","TCOM","TCON","TCPC","TCRR","TCRT","TDUP","TELA","TELZ",
        "TENB","TENS","TERN","TESS","TFFP","TFII","TFSL","TGLS","TGNA",
        "TGTX","THCA","THCP","THMA","THMO","THOR","THRD","THRM","THRY",
        "THTX","TIGR","TILE","TIPT","TIRX","TISI","TITN","TIXT","TKNO",
        "TLGA","TLMD","TLPH","TLRY","TMCI","TMDI","TMDX","TNXP","TOCA",
        "TOMZ","TOPS","TORO","TORR","TOST","TOUR","TOWN","TPCO","TPIC",
        "TPST","TPVG","TRCA","TRCB","TRDA","TREE","TRHC","TRIB","TRIN",
        "TRIT","TRMK","TRMR","TRMT","TRNS","TROV","TRTN","TRTX","TRUP",
        "TRVG","TRVN","TRVI","TRVS","TSBK","TSCO","TSHA","TSIB","TSME",
        "TSNA","TTCF","TTGT","TTMI","TTOO","TTSH","TUEM","TULM","TUMR",
        "TURN","TUSK","TUYA","TVGN","TVTX","TWKS","TWOA","TWOU","TWST",
        "TXMD","TXRH","TYME","TYRA","UBCP","UBFO","UBOH","UBSI","UCBI",
        "UCTT","UDMY","UEIC","UHLN","UIHC","ULBI","ULCC","ULGT","ULTD",
        "UMBF","UMRX","UNFI","UNFY","UNIQ","UNIT","UNJX","UNTC","UONE",
        "UONEK","UPLD","UPST","UPWK","URBN","URGN","UROY","USAS","USAU",
        "USBI","USCT","USEA","USEG","USFD","USIO","USLM","USNA","USPH",
        "UTHR","UTRS","UTSI","UUUU","UVSP","VABK","VACS","VALE","VALU",
        "VAPO","VAXX","VBFC","VBIV","VBLT","VBTX","VCFN","VCNX","VCSA",
        "VCYT","VECO","VEDU","VEEV","VERA","VERB","VERI","VERO","VERY",
        "VIAV","VICP","VICR","VIGL","VINE","VINO","VINP","VIOT","VISI",
        "VISN","VITL","VITX","VIVE","VIVK","VIVO","VJET","VLCN","VLDX",
        "VLGEA","VLIT","VLNS","VLON","VLRS","VMAR","VNDA","VNET","VNGE",
        "VNRX","VNTG","VNUE","VOXX","VOYA","VPCO","VPCB","VRAR","VRAX",
        "VRCA","VREX","VRME","VRNA","VRNS","VRNT","VRPX","VRRM","VRTA",
        "VRTV","VSCO","VSEC","VSTM","VSTN","VTAK","VTEX","VTGN","VTNA",
        "VTNR","VTOL","VTRA","VTRS","VTVT","VUZI","VVUS","VWTR","VXRT",
        "VYGR","VYNE","WABC","WAFD","WAFU","WATT","WAVD","WCLD","WDFC",
        "WEAV","WEBR","WETF","WFCF","WFRD","WGMI","WILL","WILC","WIMI",
        "WINC","WINT","WISA","WKHS","WKME","WLDN","WLFC","WLMS","WMCR",
        "WNEB","WODI","WOOF","WORX","WPRT","WRBY","WRES","WRTC","WSBC",
        "WSBF","WSFS","WSMT","WSTG","WTBA","WTFC","WTRE","WTRG","WTRH",
        "WULF","WVFC","WWBI","XBIO","XBIT","XCUR","XELA","XELB","XENE",
        "XERS","XFOR","XHER","XNCR","XNET","XOMA","XPEL","XPER","XPEV",
        "XPOF","XPON","XREG","XSPA","XTKG","YMAB","YORW","YOSH","YTEN",
        "ZAPP","ZBIO","ZCMD","ZDGE","ZEAL","ZEUS","ZFOX","ZGEN","ZGNA",
        "ZING","ZIVO","ZKIN","ZLAB","ZLDH","ZNTL","ZOTA","ZPTA","ZROZ",
        "ZTAI","ZTEK","ZTON","ZUUS","ZVRA","ZYME","ZYXI",
    ]

    def __init__(self, spark, base_path,
                 tickers=None,
                 start_date="1993-01-01",
                 checkpoint_path="/tmp/russell2000_checkpoint.parquet"):
        self.spark           = spark
        self.base_path       = base_path
        self.tickers         = tickers or self.RUSSELL2000_TICKERS
        self.start_date      = start_date
        self.checkpoint_path = checkpoint_path
        self.path            = f"{base_path}/bronze/delta/russell2000"
        self.failed          = []

        print(f"BronzeRussell2000Ingestion ✓")
        print(f"  Tickers    : {len(self.tickers)}")
        print(f"  Start date : {self.start_date}")
        print(f"  Checkpoint : {self.checkpoint_path}")
        print(f"  Output     : {self.path}")
        print(f"  Est. time  : ~25-35 minutes")

    # ------------------------------------------------------------------ #
    #  Fetch
    # ------------------------------------------------------------------ #
    def _fetch_ticker(self, ticker: str) -> pd.DataFrame:
        try:
            df = yf.download(
                ticker,
                start       = self.start_date,
                end         = datetime.today().strftime("%Y-%m-%d"),
                interval    = "1d",
                progress    = False,
                auto_adjust = True
            )

            if df is None or df.empty:
                self.failed.append(ticker)
                return pd.DataFrame()

            # Flatten MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Fix duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Normalize date column
            if "date" not in df.columns:
                for col in ["datetime","index","level_0"]:
                    if col in df.columns:
                        df = df.rename(columns={col: "date"})
                        break

            if "date" not in df.columns:
                self.failed.append(ticker)
                return pd.DataFrame()

            df["ticker"] = ticker
            df["date"]   = pd.to_datetime(df["date"]).dt.date
            df["year"]   = pd.to_datetime(df["date"]).dt.year
            df["month"]  = pd.to_datetime(df["date"]).dt.month

            keep = ["date","year","month","ticker",
                    "open","high","low","close","volume"]
            df   = df[[c for c in keep if c in df.columns]]
            df   = df.dropna(subset=["close"])
            df   = df.drop_duplicates(
                subset=["date","ticker"], keep="last"
            )
            return df

        except Exception:
            self.failed.append(ticker)
            return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        print(f"\nFetching {len(self.tickers)} tickers...")
        print(f"  Period: {self.start_date} → today")
        all_frames = []
        start      = time.time()

        for i, ticker in enumerate(self.tickers):
            if i % 100 == 0:
                elapsed = (time.time() - start) / 60
                print(f"  Progress : {i:,}/{len(self.tickers):,} "
                      f"| {elapsed:.1f}min "
                      f"| {len(all_frames):,} successful")

            df = self._fetch_ticker(ticker)
            if not df.empty:
                all_frames.append(df)

            if i % 20 == 19:
                time.sleep(0.3)

        if not all_frames:
            raise ValueError("No data fetched")

        combined = pd.concat(all_frames, ignore_index=True)
        elapsed  = (time.time() - start) / 60

        print(f"\nFetch complete in {elapsed:.1f} min:")
        print(f"  Total rows : {len(combined):,}")
        print(f"  Tickers    : {combined['ticker'].nunique():,}")
        print(f"  Failed     : {len(self.failed):,}")
        return combined

    # ------------------------------------------------------------------ #
    #  Checkpoint
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, pdf: pd.DataFrame) -> None:
        pdf.to_parquet(self.checkpoint_path, index=False)
        size = os.path.getsize(self.checkpoint_path) / 1e6
        print(f"Checkpoint saved ✓")
        print(f"  Path : {self.checkpoint_path}")
        print(f"  Size : {size:.1f} MB")
        print(f"  Rows : {len(pdf):,}")

    def load_checkpoint(self) -> pd.DataFrame:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint at {self.checkpoint_path}"
            )
        pdf = pd.read_parquet(self.checkpoint_path)
        print(f"Checkpoint loaded ✓")
        print(f"  Rows    : {len(pdf):,}")
        print(f"  Tickers : {pdf['ticker'].nunique():,}")
        return pdf

    # ------------------------------------------------------------------ #
    #  Spark
    # ------------------------------------------------------------------ #
    def _to_spark(self, pdf: pd.DataFrame):
        pdf = pdf.copy()
        pdf["date"]   = pd.to_datetime(pdf["date"]).dt.date
        pdf["year"]   = pdf["year"].astype(int)
        pdf["month"]  = pdf["month"].astype(int)
        pdf["ticker"] = pdf["ticker"].astype(str)
        pdf["open"]   = pd.to_numeric(
            pdf["open"],   errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["high"]   = pd.to_numeric(
            pdf["high"],   errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["low"]    = pd.to_numeric(
            pdf["low"],    errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["close"]  = pd.to_numeric(
            pdf["close"],  errors="coerce"
        ).fillna(0.0).astype(float)
        pdf["volume"] = pd.to_numeric(
            pdf["volume"], errors="coerce"
        ).fillna(0).astype(int)

        schema = StructType([
            StructField("date",   DateType(),    False),
            StructField("year",   IntegerType(), False),
            StructField("month",  IntegerType(), False),
            StructField("ticker", StringType(),  False),
            StructField("open",   DoubleType(),  True),
            StructField("high",   DoubleType(),  True),
            StructField("low",    DoubleType(),  True),
            StructField("close",  DoubleType(),  True),
            StructField("volume", LongType(),    True),
        ])
        return self.spark.createDataFrame(pdf, schema=schema)

    # ------------------------------------------------------------------ #
    #  Write + Optimize
    # ------------------------------------------------------------------ #
    def write_delta(self, sdf) -> None:
        print(f"\nWriting Delta: {self.path}")
        (sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema",                  "true")
            .option("delta.autoOptimize.optimizeWrite", "true")
            .option("delta.autoOptimize.autoCompact",   "true")
            .partitionBy("year", "month")
            .save(self.path)
        )
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        print("Write complete ✓")

    def optimize(self) -> None:
        print("\nOPTIMIZE + VACUUM...")
        self.spark.sql(f"OPTIMIZE delta.`{self.path}`")
        self.spark.conf.set(
            "spark.databricks.delta.retentionDurationCheck.enabled",
            "false"
        )
        self.spark.sql(
            f"VACUUM delta.`{self.path}` RETAIN 168 HOURS"
        )
        details = self.spark.sql(
            f"DESCRIBE DETAIL delta.`{self.path}`"
        ).select("numFiles","sizeInBytes").collect()[0]
        print(f"  Files : {details['numFiles']}")
        print(f"  Size  : {details['sizeInBytes']/1e6:.1f} MB")

    # ------------------------------------------------------------------ #
    #  Validate
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        print("\n" + "="*45)
        print("VALIDATION — Bronze Russell 2000")
        print("="*45)
        df    = self.spark.read.format("delta").load(self.path)
        total = df.count()
        print(f"  Total rows     : {total:,}")
        print(f"  Unique tickers : {df.select('ticker').distinct().count():,}")
        print(f"  Date range     : "
              f"{df.agg(F.min('date'), F.max('date')).collect()[0]}")
        print(f"  Null closes    : "
              f"{df.filter(F.col('close').isNull()).count()}")
        print(f"\n  Rows per year (recent):")
        df.groupBy("year").count() \
          .orderBy(F.col("year").desc()).show(10)
        print(f"\n  Sample rows:")
        df.orderBy(F.col("date").desc()) \
          .select("date","ticker","open","high",
                  "low","close","volume") \
          .show(5)
        assert total > 0, "FAIL — empty table"
        print("\nValidation PASSED ✓")

    # ------------------------------------------------------------------ #
    #  Run — with checkpoint protection
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("="*45)
        print("Bronze Russell 2000 Pipeline")
        print("="*45)

        # Step 1 — Fetch
        pdf = self.fetch_all()

        # Step 2 — Save checkpoint IMMEDIATELY after fetch
        self.save_checkpoint(pdf)

        # Step 3 — Write to Delta
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze Russell 2000 COMPLETE ✓")

    def run_from_checkpoint(self) -> None:
        """Use this if session expired after fetch but before write."""
        print("="*45)
        print("Bronze Russell 2000 — Recover from Checkpoint")
        print("="*45)
        pdf = self.load_checkpoint()
        sdf = self._to_spark(pdf)
        self.write_delta(sdf)
        self.optimize()
        self.validate()
        print("\nBronze Russell 2000 COMPLETE ✓")

# COMMAND ----------

# Re-authenticate first, then run this cell only
ingestion = BronzeRussell2000Ingestion(
    spark            = spark,
    base_path        = BASE_PATH,
    start_date       = "1993-01-01",
    checkpoint_path  = CHECKPOINT_PATH
)

ingestion.run()

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{BASE_PATH}/bronze/delta/russell2000"
)

print(f"Total rows     : {df.count():,}")
print(f"Unique tickers : {df.select('ticker').distinct().count()}")
print(f"Date range     : {df.agg(F.min('date'), F.max('date')).collect()[0]}")

df.groupBy("year").count().orderBy("year").show(10)

# COMMAND ----------

df = spark.read.format("delta").load(
    f"{BASE_PATH}/bronze/delta/russell2000"
)

df.orderBy(F.col("date").desc()) \
  .select("date","ticker","open","high","low","close","volume") \
  .show(5)