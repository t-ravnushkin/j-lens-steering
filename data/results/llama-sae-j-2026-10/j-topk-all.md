# All stored J reconstruction token components

27 SAE features; 674 positive terms (26 supports of 25; one of 24). Sorted by descending `coef_unit`. These are the COMPLETE stored sparse supports after 25 pursuit updates, not a new top-k over the full vocabulary.

Token text is a JSON string: initial spaces and escapes such as \n are meaningful. Single-token replacement characters, if any, are flagged in JSON/CSV; raw vocabulary pieces are preserved there too. Weights are coefficients on unit J atoms, not probabilities or safety contributions. No inference, new decomposition, deletion or Qwen relabeling.

Tokenizer: `RedHatAI/Llama-3.1-8B-Instruct@83c927477af31b2c66de4c0210ae5fa04b28289b`; tokenizer.json SHA256 `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`. Geometry SHA256 `4811b44b4743167157384d50a3465fa714b674ee23a3de2a6e3328707ee5d03a`. All 135 previously retained top-five token-ID/text/coefficient triples match exactly.

## Feature 1

Support: 25; reconstruction norm: 0.1804160748547546; cosine: 0.17726586825513893.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|105150|<code>"破"</code>|0.0447896272|
|2|53754|<code>" escaping"</code>|0.0434223302|
|3|57970|<code>"----\n\n"</code>|0.0351771042|
|4|12196|<code>"/."</code>|0.0347832069|
|5|63266|<code>" neatly"</code>|0.0320989601|
|6|102373|<code>"&#124;:"</code>|0.0305313542|
|7|82112|<code>"(\"~"</code>|0.0287026297|
|8|8272|<code>" followed"</code>|0.0259735025|
|9|3816|<code>" Red"</code>|0.0250740759|
|10|26616|<code>"_like"</code>|0.0237032007|
|11|15339|<code>"hello"</code>|0.0223099757|
|12|22306|<code>" inserted"</code>|0.0222839843|
|13|30120|<code>" delimiter"</code>|0.0215678811|
|14|97217|<code>"(Collections"</code>|0.0211339444|
|15|20438|<code>" tissue"</code>|0.0205825921|
|16|89063|<code>"{}\\"</code>|0.0203869641|
|17|30867|<code>"开"</code>|0.0199616961|
|18|12159|<code>"(){"</code>|0.0192191433|
|19|83058|<code>"ướ"</code>|0.0190986730|
|20|40029|<code>" brackets"</code>|0.0181929916|
|21|34824|<code>" indicators"</code>|0.0170661118|
|22|36433|<code>" homemade"</code>|0.0142298685|
|23|79341|<code>"('');\n\n"</code>|0.0140415346|
|24|284|<code>" ="</code>|0.0103202639|
|25|119101|<code>"，使"</code>|0.0060001849|

## Feature 2

Support: 25; reconstruction norm: 0.19153854600528544; cosine: 0.19014803955460263.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|18528|<code>" composition"</code>|0.0619228557|
|2|14454|<code>" formed"</code>|0.0530055314|
|3|46338|<code>" comprising"</code>|0.0351081155|
|4|105494|<code>"部分"</code>|0.0351078920|
|5|95964|<code>" tempList"</code>|0.0306019261|
|6|46847|<code>"Ingredients"</code>|0.0300839171|
|7|6090|<code>" \"%"</code>|0.0294291116|
|8|38813|<code>" {})"</code>|0.0270763915|
|9|56163|<code>"„"</code>|0.0254464801|
|10|17228|<code>"353"</code>|0.0244946200|
|11|5879|<code>"­"</code>|0.0243302826|
|12|103214|<code>"組"</code>|0.0230066348|
|13|12250|<code>" UP"</code>|0.0229362845|
|14|3525|<code>"[$"</code>|0.0207552537|
|15|24069|<code>" aggregate"</code>|0.0194927007|
|16|36295|<code>" matrices"</code>|0.0192893092|
|17|89859|<code>" Bloss"</code>|0.0190887526|
|18|67251|<code>" Hale"</code>|0.0187518392|
|19|9077|<code>" vill"</code>|0.0185054336|
|20|29446|<code>"_async"</code>|0.0171807539|
|21|75893|<code>" anatom"</code>|0.0144451410|
|22|122510|<code>" 件"</code>|0.0140673155|
|23|116326|<code>"」（"</code>|0.0133506469|
|24|1630|<code>" X"</code>|0.0108257607|
|25|95543|<code>"反"</code>|0.0061760028|

## Feature 3

Support: 25; reconstruction norm: 0.17818692176968987; cosine: 0.1787449233306517.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|423|<code>" D"</code>|0.0710810497|
|2|59214|<code>" reversal"</code>|0.0369265825|
|3|84185|<code>" reluctantly"</code>|0.0367600992|
|4|80753|<code>" sabotage"</code>|0.0334533267|
|5|84905|<code>" beware"</code>|0.0306106620|
|6|40148|<code>" villain"</code>|0.0299709085|
|7|49743|<code>"!),"</code>|0.0298779756|
|8|126200|<code>" खतर"</code>|0.0285142735|
|9|24631|<code>"\tmemset"</code>|0.0270124953|
|10|64719|<code>"Removing"</code>|0.0242088884|
|11|9595|<code>"Failed"</code>|0.0237574745|
|12|40130|<code>" darker"</code>|0.0235464163|
|13|16399|<code>" defensive"</code>|0.0232233573|
|14|7826|<code>"-B"</code>|0.0217758324|
|15|66767|<code>"(-("</code>|0.0209119506|
|16|87417|<code>" subtly"</code>|0.0202676840|
|17|88856|<code>" btnCancel"</code>|0.0198908206|
|18|114626|<code>"最後"</code>|0.0191083532|
|19|84938|<code>" newRow"</code>|0.0190038849|
|20|55200|<code>" purge"</code>|0.0189119652|
|21|98023|<code>" ownerId"</code>|0.0176002514|
|22|72836|<code>" salvage"</code>|0.0140444078|
|23|119627|<code>"ですが"</code>|0.0111966776|
|24|68735|<code>" disob"</code>|0.0082118530|
|25|480|<code>" G"</code>|0.0061554871|

## Feature 4

Support: 25; reconstruction norm: 0.22800239446696774; cosine: 0.22753117159319564.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|24271|<code>" Questions"</code>|0.0991655514|
|2|387|<code>" be"</code>|0.0746701956|
|3|17642|<code>"!\"\n\n"</code>|0.0396663360|
|4|12|<code>"-"</code>|0.0385432765|
|5|103899|<code>"答"</code>|0.0355464891|
|6|65037|<code>" keyValue"</code>|0.0351505913|
|7|86206|<code>"需要"</code>|0.0344282985|
|8|1229|<code>" Q"</code>|0.0312120039|
|9|10974|<code>"(query"</code>|0.0310171749|
|10|20126|<code>" queries"</code>|0.0304632206|
|11|119317|<code>"想要"</code>|0.0299187079|
|12|617|<code>" have"</code>|0.0276351664|
|13|121591|<code>"，或"</code>|0.0256552342|
|14|120726|<code>"有的"</code>|0.0254426971|
|15|75645|<code>"('?"</code>|0.0236160755|
|16|4860|<code>" questions"</code>|0.0211183242|
|17|10371|<code>" asking"</code>|0.0209279265|
|18|95964|<code>" tempList"</code>|0.0192349683|
|19|87093|<code>" //================================================================"</code>|0.0147092715|
|20|109461|<code>"尚"</code>|0.0118483501|
|21|43131|<code>" inclined"</code>|0.0107194297|
|22|12201|<code>" Jones"</code>|0.0089206006|
|23|61178|<code>" errMsg"</code>|0.0054831235|
|24|256|<code>"  "</code>|0.0040780199|
|25|69584|<code>"-have"</code>|0.0033148529|

## Feature 5

Support: 25; reconstruction norm: 0.26723253833665006; cosine: 0.2660961092387375.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|4028|<code>" across"</code>|0.0870734900|
|2|15603|<code>" worldwide"</code>|0.0856195018|
|3|5064|<code>" industry"</code>|0.0661756694|
|4|104514|<code>"因为"</code>|0.0579758622|
|5|6957|<code>" throughout"</code>|0.0504612327|
|6|77952|<code>" (&gt;"</code>|0.0375720970|
|7|3382|<code>"\";\n\n"</code>|0.0331101939|
|8|31663|<code>" Kan"</code>|0.0314756371|
|9|112599|<code>"現在"</code>|0.0301788170|
|10|24065|<code>"_tensor"</code>|0.0291609708|
|11|104367|<code>"ในการ"</code>|0.0286861062|
|12|113803|<code>" التج"</code>|0.0256237909|
|13|124779|<code>" благодаря"</code>|0.0250503998|
|14|65037|<code>" keyValue"</code>|0.0247454327|
|15|5156|<code>"_model"</code>|0.0246963166|
|16|6974|<code>"&gt;::"</code>|0.0238689370|
|17|122589|<code>"，因"</code>|0.0236958787|
|18|61245|<code>" Hep"</code>|0.0211745072|
|19|13882|<code>" widely"</code>|0.0208663438|
|20|64953|<code>"(\"&gt;"</code>|0.0182570089|
|21|29106|<code>" dental"</code>|0.0173358005|
|22|103786|<code>"且"</code>|0.0163451973|
|23|1462|<code>"._"</code>|0.0155232027|
|24|75529|<code>"_decimal"</code>|0.0100831678|
|25|14727|<code>" Imp"</code>|0.0083349496|

## Feature 6

Support: 24; reconstruction norm: 0.34273488622268383; cosine: 0.33761452647234763.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|1887|<code>" system"</code>|0.1380187571|
|2|3241|<code>" software"</code>|0.0727436319|
|3|37950|<code>"Implementation"</code>|0.0717482939|
|4|41441|<code>" CRM"</code>|0.0715732872|
|5|54952|<code>" integrating"</code>|0.0613993779|
|6|10487|<code>" installed"</code>|0.0483804084|
|7|58983|<code>" centralized"</code>|0.0466811061|
|8|117282|<code>"となる"</code>|0.0417588465|
|9|22293|<code>" Records"</code>|0.0336124599|
|10|86126|<code>"[System"</code>|0.0325585231|
|11|114914|<code>"解决"</code>|0.0295541361|
|12|8871|<code>" IT"</code>|0.0274208877|
|13|55292|<code>"-enabled"</code>|0.0265639946|
|14|22137|<code>" Oracle"</code>|0.0260767713|
|15|40452|<code>"管理"</code>|0.0256979410|
|16|88994|<code>" {});\n\n"</code>|0.0247898325|
|17|28598|<code>" automated"</code>|0.0228697993|
|18|17015|<code>"Asset"</code>|0.0218380988|
|19|80511|<code>":{}"</code>|0.0209725164|
|20|122765|<code>"あげ"</code>|0.0165801831|
|21|76649|<code>" bespoke"</code>|0.0155792739|
|22|127042|<code>" 통합"</code>|0.0146207698|
|23|73548|<code>"系统"</code>|0.0118755046|
|24|11798|<code>" implemented"</code>|0.0090310425|

## Feature 7

Support: 25; reconstruction norm: 0.24141231385794226; cosine: 0.23785105346505617.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|4860|<code>" questions"</code>|0.1279442012|
|2|111912|<code>"关于"</code>|0.0718611181|
|3|369|<code>" for"</code>|0.0660956725|
|4|84938|<code>" newRow"</code>|0.0345790014|
|5|7302|<code>" Phil"</code>|0.0344698578|
|6|8041|<code>" permission"</code>|0.0320544057|
|7|3249|<code>" why"</code>|0.0313036963|
|8|7887|<code>"):\n\n"</code>|0.0274775308|
|9|757|<code>" me"</code>|0.0272038709|
|10|101733|<code>"θη"</code>|0.0263676122|
|11|29486|<code>" rhet"</code>|0.0236932244|
|12|15654|<code>".impl"</code>|0.0223830845|
|13|11951|<code>" specifically"</code>|0.0215472709|
|14|61574|<code>"Reddit"</code>|0.0211296417|
|15|118256|<code>"，却"</code>|0.0194436796|
|16|25|<code>":"</code>|0.0189146474|
|17|365|<code>"ew"</code>|0.0184706878|
|18|53150|<code>" forgiveness"</code>|0.0177025460|
|19|94405|<code>"Aligned"</code>|0.0175077878|
|20|70961|<code>" requestId"</code>|0.0174086839|
|21|54107|<code>"/request"</code>|0.0169273745|
|22|123025|<code>"相同"</code>|0.0124933431|
|23|922|<code>" about"</code>|0.0106196273|
|24|75359|<code>" newObj"</code>|0.0069252793|
|25|95659|<code>" sequentially"</code>|0.0050991387|

## Feature 9

Support: 25; reconstruction norm: 0.3116359843594489; cosine: 0.30907659226948764.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|46672|<code>" specializes"</code>|0.1071614921|
|2|37048|<code>" renowned"</code>|0.0957817212|
|3|10968|<code>" focused"</code>|0.0625055283|
|4|43111|<code>" dynamically"</code>|0.0444825627|
|5|80068|<code>"Manufact"</code>|0.0442395061|
|6|36283|<code>" Prim"</code>|0.0404420272|
|7|71674|<code>" pioneering"</code>|0.0391885675|
|8|89642|<code>" '..',"</code>|0.0382741354|
|9|17843|<code>" producing"</code>|0.0333000757|
|10|107226|<code>"是一"</code>|0.0328861214|
|11|61236|<code>" Cater"</code>|0.0328290947|
|12|118229|<code>"经营"</code>|0.0325256512|
|13|119818|<code>" الأح"</code>|0.0310607944|
|14|2625|<code>"(["</code>|0.0304152519|
|15|77274|<code>" minX"</code>|0.0283980947|
|16|35410|<code>" pharmaceutical"</code>|0.0280426480|
|17|124038|<code>" الصن"</code>|0.0264757015|
|18|85957|<code>" diversified"</code>|0.0259889793|
|19|15355|<code>"�"</code>|0.0246461015|
|20|15871|<code>" primarily"</code>|0.0223084297|
|21|56034|<code>"((-"</code>|0.0216921996|
|22|71010|<code>" encompasses"</code>|0.0160142034|
|23|113368|<code>"负责"</code>|0.0123967808|
|24|57114|<code>"_\\"</code>|0.0085619325|
|25|63257|<code>"focused"</code>|0.0048964680|

## Feature 10

Support: 25; reconstruction norm: 0.18516815650106833; cosine: 0.18468130200158334.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|92206|<code>"/from"</code>|0.0736028403|
|2|28157|<code>" converter"</code>|0.0388446786|
|3|2352|<code>"aster"</code>|0.0387959927|
|4|95964|<code>" tempList"</code>|0.0343884602|
|5|27020|<code>")_"</code>|0.0328047350|
|6|735|<code>" K"</code>|0.0324936248|
|7|8903|<code>" reality"</code>|0.0319115631|
|8|39841|<code>" Speech"</code>|0.0258940142|
|9|2469|<code>"ots"</code>|0.0255959332|
|10|95119|<code>" unwitting"</code>|0.0229576062|
|11|94315|<code>" overflowing"</code>|0.0220294166|
|12|30591|<code>"あ"</code>|0.0190230180|
|13|260|<code>"        "</code>|0.0187683161|
|14|71128|<code>" manageable"</code>|0.0181634724|
|15|27084|<code>" subset"</code>|0.0179035962|
|16|85248|<code>" trope"</code>|0.0176295750|
|17|83532|<code>"&#124;&#124;("</code>|0.0175592564|
|18|2991|<code>" Text"</code>|0.0168822426|
|19|14747|<code>" conversion"</code>|0.0158121418|
|20|19306|<code>"930"</code>|0.0153694386|
|21|78872|<code>" mRNA"</code>|0.0151309697|
|22|13021|<code>" mapping"</code>|0.0142005635|
|23|25765|<code>"!\"\n"</code>|0.0140890675|
|24|3522|<code>"gether"</code>|0.0113490075|
|25|113454|<code>"嘴"</code>|0.0074874768|

## Feature 11

Support: 25; reconstruction norm: 0.34134296884205406; cosine: 0.33580694914326237.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|15592|<code>" AI"</code>|0.1376051754|
|2|65767|<code>" linguistic"</code>|0.0842731521|
|3|65273|<code>" userInput"</code>|0.0555847511|
|4|6369|<code>" chat"</code>|0.0501362644|
|5|21633|<code>" conversations"</code>|0.0456731319|
|6|16674|<code>" outputs"</code>|0.0452085473|
|7|16483|<code>" writers"</code>|0.0441362001|
|8|109632|<code>"训"</code>|0.0374199077|
|9|66251|<code>" contextual"</code>|0.0361788049|
|10|16934|<code>"Limit"</code>|0.0358396433|
|11|56725|<code>"Speech"</code>|0.0344914794|
|12|35649|<code>" personalized"</code>|0.0328309275|
|13|1217|<code>" user"</code>|0.0303419605|
|14|16681|<code>" interact"</code>|0.0303011537|
|15|30828|<code>" neural"</code>|0.0302988794|
|16|88335|<code>" iterating"</code>|0.0286364965|
|17|7457|<code>"&gt;{"</code>|0.0283124857|
|18|50712|<code>" Alexa"</code>|0.0258634686|
|19|35164|<code>" Discord"</code>|0.0253845453|
|20|1199|<code>"Text"</code>|0.0242563132|
|21|95300|<code>" Emotional"</code>|0.0212320294|
|22|16572|<code>" trained"</code>|0.0183419380|
|23|7887|<code>"):\n\n"</code>|0.0148429777|
|24|8865|<code>"-user"</code>|0.0116445543|
|25|7899|<code>" voice"</code>|0.0071661589|

## Feature 12

Support: 25; reconstruction norm: 0.3137365171876735; cosine: 0.3102296003618969.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|1288|<code>" should"</code>|0.1242716759|
|2|112255|<code>" مناسب"</code>|0.0743790865|
|3|51504|<code>"选择"</code>|0.0614221171|
|4|14791|<code>" suitable"</code>|0.0534444712|
|5|23669|<code>" optimal"</code>|0.0519971475|
|6|11349|<code>" recommended"</code>|0.0451834947|
|7|57806|<code>"Needed"</code>|0.0419070274|
|8|107585|<code>"应该"</code>|0.0395192504|
|9|11223|<code>" proposed"</code>|0.0384501927|
|10|50663|<code>" ideally"</code>|0.0350949578|
|11|117282|<code>"となる"</code>|0.0305942222|
|12|32876|<code>"Must"</code>|0.0288247205|
|13|1161|<code>"(s"</code>|0.0286247730|
|14|435|<code>" F"</code>|0.0246142633|
|15|8446|<code>" strategy"</code>|0.0244020764|
|16|20477|<code>"=?"</code>|0.0240425691|
|17|57277|<code>"**\n\n"</code>|0.0229792986|
|18|60342|<code>" Tracy"</code>|0.0228725690|
|19|122589|<code>"，因"</code>|0.0199142583|
|20|393|<code>" P"</code>|0.0178991593|
|21|124116|<code>"考虑"</code>|0.0153025454|
|22|33983|<code>" μ"</code>|0.0145878652|
|23|8670|<code>" requirements"</code>|0.0112076141|
|24|98611|<code>" gridSize"</code>|0.0100033544|
|25|27148|<code>"[\\"</code>|0.0068037491|

## Feature 13

Support: 25; reconstruction norm: 0.22678788655590273; cosine: 0.22203251563207113.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|46493|<code>"   "</code>|0.0726309866|
|2|328|<code>" S"</code>|0.0643441826|
|3|29438|<code>" Introduction"</code>|0.0456805229|
|4|3639|<code>" What"</code>|0.0345340706|
|5|3146|<code>" **"</code>|0.0317661650|
|6|415|<code>"     "</code>|0.0315867849|
|7|31439|<code>" straightforward"</code>|0.0313927345|
|8|16186|<code>"_initial"</code>|0.0293968003|
|9|1050|<code>" Re"</code>|0.0278878734|
|10|3314|<code>" State"</code>|0.0266351048|
|11|12515|<code>" ...\n"</code>|0.0261670519|
|12|33764|<code>"对"</code>|0.0252889059|
|13|836|<code>" name"</code>|0.0229811687|
|14|22654|<code>" identification"</code>|0.0219202843|
|15|29041|<code>" Kl"</code>|0.0208035056|
|16|16556|<code>"で"</code>|0.0201492663|
|17|13017|<code>"\tF"</code>|0.0196499135|
|18|30883|<code>" acute"</code>|0.0177291334|
|19|3331|<code>" General"</code>|0.0165334344|
|20|24379|<code>" nearest"</code>|0.0164860003|
|21|122098|<code>" "</code>|0.0159696545|
|22|356|<code>" C"</code>|0.0119738504|
|23|126075|<code>" 必"</code>|0.0102917049|
|24|118544|<code>"いか"</code>|0.0053366716|
|25|386|<code>" M"</code>|0.0042365650|

## Feature 14

Support: 25; reconstruction norm: 0.24493643889448538; cosine: 0.23966498189882596.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|9193|<code>" extremely"</code>|0.0548694842|
|2|549|<code>" U"</code>|0.0529538095|
|3|1054|<code>" “"</code>|0.0514170341|
|4|1877|<code>" sex"</code>|0.0446749590|
|5|1217|<code>" user"</code>|0.0418624766|
|6|5196|<code>" itself"</code>|0.0392763950|
|7|12966|<code>" humans"</code>|0.0382823721|
|8|39633|<code>" infamous"</code>|0.0349633843|
|9|6774|<code>"((("</code>|0.0335225202|
|10|14|<code>"/"</code>|0.0323607065|
|11|60277|<code>" fantasies"</code>|0.0314618014|
|12|67321|<code>"=[],"</code>|0.0295803025|
|13|58135|<code>" glor"</code>|0.0269147493|
|14|94847|<code>" ridicule"</code>|0.0267215837|
|15|3707|<code>")\n\n\n"</code>|0.0245905295|
|16|21633|<code>" conversations"</code>|0.0243733898|
|17|65273|<code>" userInput"</code>|0.0242977571|
|18|8996|<code>" whatever"</code>|0.0237161871|
|19|56438|<code>"所有"</code>|0.0229447633|
|20|119508|<code>" любой"</code>|0.0189763475|
|21|10836|<code>" Random"</code>|0.0180744845|
|22|119292|<code>"符合"</code>|0.0134285875|
|23|480|<code>" G"</code>|0.0134094711|
|24|42246|<code>"其"</code>|0.0073270695|
|25|5526|<code>" popular"</code>|0.0054593091|

## Feature 15

Support: 25; reconstruction norm: 0.22232779074289732; cosine: 0.22161961214116818.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|1303|<code>"ting"</code>|0.0991658494|
|2|11586|<code>" Pet"</code>|0.0792998075|
|3|7980|<code>" owners"</code>|0.0508747809|
|4|65245|<code>" cudd"</code>|0.0360259227|
|5|12875|<code>" dogs"</code>|0.0342669152|
|6|17|<code>"2"</code>|0.0331621617|
|7|87028|<code>" gruesome"</code>|0.0283941906|
|8|1293|<code>"erson"</code>|0.0282134060|
|9|75519|<code>" furry"</code>|0.0278227534|
|10|5364|<code>"-M"</code>|0.0258768331|
|11|69733|<code>"ulant"</code>|0.0234542079|
|12|36035|<code>" Emma"</code>|0.0233920924|
|13|91061|<code>" dataSize"</code>|0.0221860800|
|14|90149|<code>"):\\"</code>|0.0209948234|
|15|105898|<code>"きた"</code>|0.0189479198|
|16|82061|<code>"&lt;=("</code>|0.0188793577|
|17|39040|<code>" paren"</code>|0.0177468657|
|18|97137|<code>" newPath"</code>|0.0171415191|
|19|60344|<code>"_remaining"</code>|0.0160252061|
|20|31540|<code>"可"</code>|0.0151877431|
|21|39824|<code>" rabbit"</code>|0.0145304278|
|22|115205|<code>"ा।\n\n"</code>|0.0126398867|
|23|14830|<code>" clinical"</code>|0.0121803312|
|24|102376|<code>" люб"</code>|0.0077367504|
|25|57482|<code>" reliably"</code>|0.0071458332|

## Feature 16

Support: 25; reconstruction norm: 0.41257752252450924; cosine: 0.416385807897089.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|2536|<code>" non"</code>|0.1030247509|
|2|8389|<code>" negative"</code>|0.0687700212|
|3|111208|<code>" عدم"</code>|0.0679622069|
|4|4276|<code>" NOT"</code>|0.0626335740|
|5|19821|<code>" absence"</code>|0.0548987947|
|6|3660|<code>"(false"</code>|0.0546382554|
|7|38001|<code>" rejection"</code>|0.0488307029|
|8|28544|<code>" excluded"</code>|0.0488260165|
|9|42257|<code>"&gt;No"</code>|0.0482005179|
|10|82802|<code>" negligible"</code>|0.0395668261|
|11|33476|<code>"否"</code>|0.0395538807|
|12|47591|<code>" denies"</code>|0.0391965471|
|13|70284|<code>"不存在"</code>|0.0391904004|
|14|45497|<code>" assertFalse"</code>|0.0380278789|
|15|32588|<code>"_none"</code>|0.0341596901|
|16|15|<code>"0"</code>|0.0321327858|
|17|86819|<code>" Nope"</code>|0.0293732006|
|18|104435|<code>"なかった"</code>|0.0283291936|
|19|127208|<code>"，没有"</code>|0.0266320221|
|20|27156|<code>" Buck"</code>|0.0229125340|
|21|510|<code>" ["</code>|0.0207699817|
|22|35873|<code>"._\n\n"</code>|0.0194142386|
|23|31941|<code>" Dry"</code>|0.0173399560|
|24|117492|<code>" отри"</code>|0.0114194062|
|25|13738|<code>" dismiss"</code>|0.0096488595|

## Feature 17

Support: 25; reconstruction norm: 0.24314128729485718; cosine: 0.23903829012129502.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|7036|<code>" scope"</code>|0.0713739619|
|2|3392|<code>" amount"</code>|0.0512359887|
|3|23246|<code>" (&lt;"</code>|0.0479731560|
|4|18539|<code>" availability"</code>|0.0429067425|
|5|19439|<code>" Limited"</code>|0.0394167416|
|6|5070|<code>" resources"</code>|0.0349951796|
|7|94840|<code>" scalability"</code>|0.0331313424|
|8|108084|<code>" времени"</code>|0.0323164426|
|9|2680|<code>" access"</code>|0.0321138501|
|10|18240|<code>" \\("</code>|0.0300356410|
|11|67016|<code>"しか"</code>|0.0297958069|
|12|36317|<code>"/un"</code>|0.0276945960|
|13|99442|<code>" startPosition"</code>|0.0270479303|
|14|110229|<code>"ところ"</code>|0.0259268768|
|15|65483|<code>"-duration"</code>|0.0253146999|
|16|14711|<code>"###"</code>|0.0211102981|
|17|78223|<code>" expressive"</code>|0.0206052810|
|18|17735|<code>"503"</code>|0.0195203274|
|19|42034|<code>" Palestine"</code>|0.0195032861|
|20|124533|<code>"胆"</code>|0.0189686362|
|21|11162|<code>" DC"</code>|0.0174424145|
|22|23705|<code>" startTime"</code>|0.0158211086|
|23|24035|<code>" visibility"</code>|0.0143187642|
|24|61742|<code>" Liam"</code>|0.0109960353|
|25|10736|<code>" funds"</code>|0.0070499480|

## Feature 18

Support: 25; reconstruction norm: 0.25397160556612897; cosine: 0.25488934741506253.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|304|<code>" in"</code>|0.1067932770|
|2|38916|<code>" Born"</code>|0.0669724867|
|3|6250|<code>" September"</code>|0.0657605678|
|4|21990|<code>"生"</code>|0.0468616858|
|5|35304|<code>"于"</code>|0.0410740711|
|6|51764|<code>" Frederick"</code>|0.0367628559|
|7|1535|<code>"(),"</code>|0.0335880108|
|8|110856|<code>"ในป"</code>|0.0333427675|
|9|23327|<code>".timestamp"</code>|0.0312508084|
|10|121300|<code>"ولد"</code>|0.0295577161|
|11|41769|<code>" Ahmed"</code>|0.0291970968|
|12|127208|<code>"，没有"</code>|0.0287222546|
|13|53208|<code>"[{"</code>|0.0284205303|
|14|71058|<code>" Carmen"</code>|0.0280439463|
|15|7342|<code>" birth"</code>|0.0279948842|
|16|88156|<code>" paramMap"</code>|0.0244006608|
|17|45918|<code>"�"</code>|0.0236535575|
|18|89019|<code>" prematurely"</code>|0.0226415396|
|19|19809|<code>" trailer"</code>|0.0220701229|
|20|84938|<code>" newRow"</code>|0.0193483587|
|21|24552|<code>" centuries"</code>|0.0175825283|
|22|92489|<code>"(\"**"</code>|0.0148026580|
|23|53182|<code>" Stockholm"</code>|0.0143118864|
|24|35476|<code>"{-#"</code>|0.0091217235|
|25|17|<code>"2"</code>|0.0081489775|

## Feature 19

Support: 25; reconstruction norm: 0.2674594327867056; cosine: 0.2637599840280756.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|4820|<code>" parse"</code>|0.0957929492|
|2|16489|<code>" converted"</code>|0.0872455239|
|3|13187|<code>" parseInt"</code>|0.0722057298|
|4|17|<code>"2"</code>|0.0420104638|
|5|16972|<code>"=int"</code>|0.0419612005|
|6|5720|<code>" eval"</code>|0.0410709232|
|7|63013|<code>" sscanf"</code>|0.0384200662|
|8|11514|<code>" Cast"</code>|0.0344304591|
|9|45591|<code>"FromString"</code>|0.0330122449|
|10|35876|<code>" numerical"</code>|0.0296173058|
|11|4560|<code>" trying"</code>|0.0285988059|
|12|17642|<code>"!\"\n\n"</code>|0.0232942533|
|13|4736|<code>".parse"</code>|0.0229930356|
|14|4172|<code>"(-"</code>|0.0218082126|
|15|27562|<code>" Nelson"</code>|0.0204358175|
|16|87599|<code>" BJ"</code>|0.0197039694|
|17|67396|<code>" tonumber"</code>|0.0193159059|
|18|94532|<code>" hastily"</code>|0.0143623054|
|19|26293|<code>" atoi"</code>|0.0136035746|
|20|4194|<code>" "</code>|0.0125081912|
|21|8432|<code>"(float"</code>|0.0123585593|
|22|64695|<code>" Peach"</code>|0.0119517324|
|23|83687|<code>"数字"</code>|0.0100771226|
|24|510|<code>" ["</code>|0.0083430856|
|25|33432|<code>" validated"</code>|0.0047698864|

## Feature 21

Support: 25; reconstruction norm: 0.42479108291591405; cosine: 0.42341324527939517.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|5845|<code>" ability"</code>|0.2058230340|
|2|27327|<code>"能"</code>|0.1116720736|
|3|13171|<code>" capable"</code>|0.0940592363|
|4|76588|<code>" adept"</code>|0.0629750490|
|5|31869|<code>" understands"</code>|0.0481412709|
|6|84938|<code>" newRow"</code>|0.0435223356|
|7|61440|<code>" seamlessly"</code>|0.0430449396|
|8|37059|<code>"Successfully"</code>|0.0361625813|
|9|7887|<code>"):\n\n"</code>|0.0346029624|
|10|59531|<code>" Adapt"</code>|0.0328561515|
|11|119572|<code>"懂"</code>|0.0325384401|
|12|37837|<code>" Nathan"</code>|0.0300175492|
|13|76247|<code>"Flexible"</code>|0.0275366493|
|14|39375|<code>" willingness"</code>|0.0252490975|
|15|80853|<code>"Demon"</code>|0.0241610166|
|16|106594|<code>"善"</code>|0.0228406023|
|17|7826|<code>"-B"</code>|0.0221090280|
|18|110577|<code>"既"</code>|0.0207007769|
|19|35003|<code>"Ability"</code>|0.0206418633|
|20|67975|<code>" Recogn"</code>|0.0174276680|
|21|7148|<code>"{$"</code>|0.0169167537|
|22|386|<code>" M"</code>|0.0143397218|
|23|91214|<code>" propName"</code>|0.0141099403|
|24|113961|<code>"能够"</code>|0.0079322513|
|25|7011|<code>" Mat"</code>|0.0054437458|

## Feature 22

Support: 25; reconstruction norm: 0.17731106438840724; cosine: 0.17609674612481124.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|16250|<code>" membership"</code>|0.0455146059|
|2|70003|<code>" associative"</code>|0.0433423407|
|3|32413|<code>" Partner"</code>|0.0387137569|
|4|117942|<code>" العامة"</code>|0.0308346152|
|5|449|<code>" with"</code>|0.0304803085|
|6|44125|<code>" Montgomery"</code>|0.0296209306|
|7|36802|<code>"{\\"</code>|0.0288856756|
|8|978|<code>"é"</code>|0.0261794236|
|9|81826|<code>" plentiful"</code>|0.0259186011|
|10|14561|<code>" professor"</code>|0.0258249342|
|11|52407|<code>"&lt;pair"</code>|0.0254874621|
|12|86757|<code>" arbitrarily"</code>|0.0254105218|
|13|34975|<code>" Koh"</code>|0.0245819036|
|14|105047|<code>"急"</code>|0.0243177824|
|15|110018|<code>"とき"</code>|0.0238153748|
|16|8743|<code>" PR"</code>|0.0231407713|
|17|53400|<code>"된"</code>|0.0222562701|
|18|96184|<code>" fieldValue"</code>|0.0204529911|
|19|30810|<code>"(\"\\\\"</code>|0.0179095473|
|20|19065|<code>" blame"</code>|0.0175266862|
|21|85129|<code>" disparate"</code>|0.0171640925|
|22|20263|<code>" Jean"</code>|0.0132495370|
|23|45832|<code>" '''\n\n"</code>|0.0125337113|
|24|47185|<code>"_="</code>|0.0082196640|
|25|52605|<code>" triggering"</code>|0.0060206358|

## Feature 23

Support: 25; reconstruction norm: 0.1629282214798345; cosine: 0.1612834975318575.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|1049|<code>"200"</code>|0.0477669798|
|2|61750|<code>" Malta"</code>|0.0450433083|
|3|18789|<code>"(async"</code>|0.0446925759|
|4|59246|<code>"}:{"</code>|0.0386618339|
|5|36682|<code>"(({"</code>|0.0302509256|
|6|34684|<code>" mutations"</code>|0.0260456093|
|7|314|<code>" {"</code>|0.0259470306|
|8|23634|<code>" nest"</code>|0.0214521550|
|9|64680|<code>" Midnight"</code>|0.0214165933|
|10|38942|<code>"Ax"</code>|0.0213963576|
|11|1920|<code>" process"</code>|0.0201780554|
|12|65410|<code>" cooled"</code>|0.0195848066|
|13|36489|<code>" disclosed"</code>|0.0185422190|
|14|11586|<code>" Pet"</code>|0.0178944003|
|15|84273|<code>" adorned"</code>|0.0173931681|
|16|18407|<code>" SN"</code>|0.0173898600|
|17|78182|<code>" pineapple"</code>|0.0162268840|
|18|39571|<code>" apost"</code>|0.0155320652|
|19|30558|<code>" Ki"</code>|0.0149992248|
|20|38853|<code>"`${"</code>|0.0149603803|
|21|2671|<code>" options"</code>|0.0134021631|
|22|9740|<code>"BD"</code>|0.0132805146|
|23|69477|<code>" unmatched"</code>|0.0110710915|
|24|23015|<code>"[N"</code>|0.0084266812|
|25|8743|<code>" PR"</code>|0.0065868767|

## Feature 24

Support: 25; reconstruction norm: 0.16464636002768143; cosine: 0.1616432096827353.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|13488|<code>" explore"</code>|0.0477120318|
|2|7887|<code>"):\n\n"</code>|0.0428314470|
|3|87528|<code>"']*"</code>|0.0418256521|
|4|456|<code>"get"</code>|0.0362747796|
|5|60124|<code>"&amp;&amp;("</code>|0.0334251747|
|6|15884|<code>" carefully"</code>|0.0329526328|
|7|387|<code>" be"</code>|0.0281946473|
|8|116168|<code>"ールド"</code>|0.0261275824|
|9|85518|<code>" currentPosition"</code>|0.0259365086|
|10|5034|<code>" La"</code>|0.0258411989|
|11|63652|<code>" prioritize"</code>|0.0243791435|
|12|67233|<code>" skepticism"</code>|0.0222493336|
|13|23833|<code>"782"</code>|0.0219591167|
|14|71462|<code>" extrapol"</code>|0.0214889646|
|15|83360|<code>"thinking"</code>|0.0206383988|
|16|113797|<code>"尊"</code>|0.0202026982|
|17|38248|<code>"ー�"</code>|0.0198436063|
|18|71174|<code>"声"</code>|0.0192714613|
|19|34074|<code>" adjustments"</code>|0.0187312942|
|20|12201|<code>" Jones"</code>|0.0166073423|
|21|15044|<code>"()),\n"</code>|0.0164652504|
|22|48606|<code>" Florence"</code>|0.0129181780|
|23|87185|<code>"('^"</code>|0.0108143575|
|24|1304|<code>" make"</code>|0.0064229337|
|25|108441|<code>" حيث"</code>|0.0044258991|

## Feature 25

Support: 25; reconstruction norm: 0.4432770310724381; cosine: 0.4431698785903105.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|16296|<code>" consequences"</code>|0.2472941577|
|2|114806|<code>"，如果"</code>|0.1271404624|
|3|422|<code>" if"</code>|0.1154293492|
|4|54677|<code>" catastrophic"</code>|0.0644915849|
|5|16750|<code>" penalty"</code>|0.0636708513|
|6|11366|<code>"_if"</code>|0.0625232905|
|7|25060|<code>" punishment"</code>|0.0553421602|
|8|60722|<code>"Reward"</code>|0.0514601506|
|9|445|<code>" L"</code>|0.0394555740|
|10|17965|<code>"Failure"</code>|0.0273478273|
|11|115004|<code>"损"</code>|0.0273047313|
|12|6774|<code>"((("</code>|0.0247288384|
|13|54463|<code>" Ris"</code>|0.0222824775|
|14|95285|<code>" Benny"</code>|0.0220573321|
|15|84938|<code>" newRow"</code>|0.0219464973|
|16|40265|<code>"当"</code>|0.0204679258|
|17|73113|<code>"}-{"</code>|0.0195748005|
|18|24855|<code>" Gh"</code>|0.0193823129|
|19|7826|<code>"-B"</code>|0.0190882497|
|20|55131|<code>"Scenario"</code>|0.0190199856|
|21|83098|<code>" toReturn"</code>|0.0182449706|
|22|38093|<code>"会"</code>|0.0141355414|
|23|66110|<code>" Dre"</code>|0.0126792556|
|24|109688|<code>"危"</code>|0.0080151455|
|25|32917|<code>" Columbus"</code>|0.0070313932|

## Feature 26

Support: 25; reconstruction norm: 0.3405270737038961; cosine: 0.34958497403888655.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|473|<code>" H"</code>|0.0921149030|
|2|279|<code>" the"</code>|0.0850837156|
|3|1050|<code>" Re"</code>|0.0504137576|
|4|105584|<code>" "</code>|0.0444017425|
|5|6642|<code>" setup"</code>|0.0404883064|
|6|32710|<code>"quad"</code>|0.0375592783|
|7|1891|<code>" di"</code>|0.0373002961|
|8|5422|<code>" Key"</code>|0.0362087041|
|9|11852|<code>"-level"</code>|0.0348518863|
|10|106050|<code>"  \n"</code>|0.0341438390|
|11|38632|<code>" encompass"</code>|0.0340878367|
|12|17910|<code>" characteristics"</code>|0.0301752016|
|13|6011|<code>" Department"</code>|0.0296801943|
|14|54755|<code>" Burton"</code>|0.0281592179|
|15|59739|<code>"つ"</code>|0.0276490822|
|16|18824|<code>" champion"</code>|0.0276371688|
|17|2585|<code>" control"</code>|0.0252321288|
|18|45271|<code>"]=-"</code>|0.0250683576|
|19|1038|<code>"\n\n\n\n"</code>|0.0244597327|
|20|102769|<code>"切"</code>|0.0242588818|
|21|640|<code>"aram"</code>|0.0221821498|
|22|1080|<code>" co"</code>|0.0209764950|
|23|1040|<code>"const"</code>|0.0181340519|
|24|1883|<code>" App"</code>|0.0124769025|
|25|52778|<code>"████"</code>|0.0103970375|

## Feature 27

Support: 25; reconstruction norm: 0.17271182625883155; cosine: 0.17277019514961048.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|85206|<code>" chlorine"</code>|0.0447866842|
|2|82|<code>"s"</code>|0.0430043079|
|3|9607|<code>"-D"</code>|0.0384713709|
|4|92489|<code>"(\"**"</code>|0.0384135209|
|5|64568|<code>" crystall"</code>|0.0333676040|
|6|921|<code>" Ch"</code>|0.0310383327|
|7|89761|<code>"_lost"</code>|0.0309020560|
|8|32103|<code>" Lucas"</code>|0.0271286480|
|9|307|<code>"id"</code>|0.0257029198|
|10|84938|<code>" newRow"</code>|0.0250667483|
|11|7672|<code>"(ch"</code>|0.0246548019|
|12|9761|<code>"lor"</code>|0.0240362082|
|13|71313|<code>" collagen"</code>|0.0216016769|
|14|35072|<code>" delighted"</code>|0.0210426673|
|15|24572|<code>"/-"</code>|0.0209900439|
|16|126370|<code>"，全"</code>|0.0206717290|
|17|18336|<code>" radical"</code>|0.0206117071|
|18|87128|<code>" Schwarz"</code>|0.0203002188|
|19|110456|<code>"刺"</code>|0.0197390597|
|20|14699|<code>")(("</code>|0.0172280967|
|21|82688|<code>" aromatic"</code>|0.0154808881|
|22|110482|<code>"困"</code>|0.0150237773|
|23|17|<code>"2"</code>|0.0139013464|
|24|69749|<code>"_coordinate"</code>|0.0095673585|
|25|2773|<code>" prom"</code>|0.0062994170|

## Feature 29

Support: 25; reconstruction norm: 0.27947145853095107; cosine: 0.2784457075219124.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|25|<code>":"</code>|0.2379338145|
|2|118995|<code>"ा\n\n"</code>|0.0685982630|
|3|551|<code>" :"</code>|0.0614437535|
|4|113275|<code>" عشر"</code>|0.0568167530|
|5|50479|<code>" Fury"</code>|0.0351537690|
|6|32406|<code>" Creating"</code>|0.0330538638|
|7|29829|<code>" Mari"</code>|0.0328212716|
|8|8738|<code>" continued"</code>|0.0313885547|
|9|50258|<code>"_vectors"</code>|0.0308285467|
|10|9595|<code>"Failed"</code>|0.0290677417|
|11|82367|<code>"_allocation"</code>|0.0270809345|
|12|86159|<code>".dependencies"</code>|0.0247517396|
|13|54946|<code>" poised"</code>|0.0247358717|
|14|97137|<code>" newPath"</code>|0.0244014319|
|15|124920|<code>"忆"</code>|0.0241946802|
|16|62549|<code>" Enhanced"</code>|0.0240941402|
|17|13345|<code>" Valley"</code>|0.0238027684|
|18|99683|<code>" tactile"</code>|0.0232163109|
|19|70627|<code>" pounding"</code>|0.0217349771|
|20|42779|<code>" intuitive"</code>|0.0207276512|
|21|85518|<code>" currentPosition"</code>|0.0205469541|
|22|17461|<code>"оль"</code>|0.0162202418|
|23|10323|<code>" Self"</code>|0.0148914224|
|24|5061|<code>"\\t"</code>|0.0109376898|
|25|126268|<code>"つけ"</code>|0.0104674827|

## Feature 31

Support: 25; reconstruction norm: 0.3045350464950666; cosine: 0.3113521255886499.

|Rank|Token ID|Decoded token (JSON string)|coef_unit|
|---:|---:|---|---:|
|1|70718|<code>" gigg"</code>|0.0659863800|
|2|12304|<code>" excited"</code>|0.0592176020|
|3|113261|<code>"幼"</code>|0.0546833500|
|4|54546|<code>" Daddy"</code>|0.0541973002|
|5|52335|<code>" toddler"</code>|0.0512887053|
|6|23|<code>"8"</code>|0.0501008220|
|7|47303|<code>"(children"</code>|0.0419737361|
|8|6699|<code>" parents"</code>|0.0401037820|
|9|6975|<code>" learning"</code>|0.0346783921|
|10|61914|<code>" fascinated"</code>|0.0346481875|
|11|28088|<code>" inherited"</code>|0.0331142135|
|12|49481|<code>" Toys"</code>|0.0324773379|
|13|469|<code>" E"</code>|0.0307786427|
|14|125150|<code>"哭"</code>|0.0293733776|
|15|82240|<code>" mommy"</code>|0.0285737496|
|16|68735|<code>" disob"</code>|0.0285227727|
|17|87879|<code>"!).\n\n"</code>|0.0282761212|
|18|256|<code>"  "</code>|0.0275436528|
|19|16576|<code>" Boy"</code>|0.0261118077|
|20|15042|<code>" grown"</code>|0.0244628061|
|21|15394|<code>" Children"</code>|0.0223134812|
|22|84938|<code>" newRow"</code>|0.0195181519|
|23|105318|<code>"要求"</code>|0.0168876052|
|24|45323|<code>" sibling"</code>|0.0141568100|
|25|14543|<code>":="</code>|0.0091113355|

