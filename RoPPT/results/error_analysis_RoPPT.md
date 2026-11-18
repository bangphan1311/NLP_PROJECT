# RoPPT (Bạn 4) — VUA-18 results & error analysis

## Test metrics (best-dev checkpoint, full test set)

| | Precision | Recall | F1 |
|---|---|---|---|
| **Ours** | 78.20 | 72.69 | **75.34** |
| Paper (RoPPT) | 80.0 | 78.2 | 79.1 |
| MelBERT (paper) | 80.1 | 76.9 | 78.5 |

Confusion: TP=4538 FP=1265 FN=1705 TN=36439

Gap cause: recall lower by 5.3 pts. Dev F1 still rising at last eval (0.651 -> 0.719 -> 0.744); training was 2 epochs on CPU + 1 resumed epoch on GPU with restarted LR schedule, batch 16. Not converged vs the paper's 3-epoch continuous schedule.

## 15 False Positives (predicted metaphor, gold literal)

- **['turned']** — ['liverpool', ' dockers', ' recently', ' turned', ' back', ' parts', ' of', ' a', ' 1,500', '-', 'tonne', ' consignment', ' of', ' canadian', ' pcbs', '(', ' polychlorinated', ' biphenyls', ')', ' bound', ' for', ' this', ' plant', ' in', ' gwent', '.', '\n']
- **['call']** — ['biblical', ' motifs', ' recur', ' â\x80\x94', ' some', ' art', '-', 'experts', ' call', ' them', ' sermons', ' in', ' stone', ',', ' and', ' their', ' messages', ' would', ' certainly', ' have', ' been', ' clear', ' to', ' the', ' often', '-', 'illiterate', ' medieval', ' pilgrims', ' trudging', '
- **['gave']** — ['i', ' know', ' five', ' hundred', ' and', ' sixty', ' was', ' extra', ' cos', ' stuart', ' gave', ' me', ' two', ' hundred', ' short', ' of', ' a', ' thousand', '.', '\n']
- **['to']** — ['yeah', ',', ' must', ' be', ' used', ' to', ' the', ' must', ' do', ' it', ' sometime', '\n']
- **['got']** — ['two', ' months', ' and', ' she', ' phoned', ' up', ' and', ' i', ' kind', ' of', ' made', ' out', ' it', ' got', ' lost', ' in', ' the', ' post', '.', '\n']
- **['down']** — ['buddy', ' invites', ' us', ' to', ' regard', ' holly', ' in', ' the', ' same', ' light', ',', ' mocking', '(', ' in', ' a', ' horribly', ' nudging', ' re', '-', 'creation', ')', ' lubbock', ' old', '-', 'timers', ' like', ' the', ' hayriders', ',', ' who', ' sing', ':', ' â\x80\x98', ' when', ' th
- **['headed']** — ['the', ' treasury', ' secretary', ' nicholas', ' brady', ',', ' a', ' level', ' headed', ' figure', ',', ' warned', ' that', ' the', ' imbroglio', ' may', ' lead', ' to', ' higher', ' interest', ' costs', ' for', ' the', ' taxpayer', '.', '\n']
- **['place']** — ['it', ' is', ' simple', ' enough', ' for', ' him', ' to', ' show', ' that', ' a', ' boxing', ' match', ' takes', ' place', ' within', ' a', ' general', ' totality', ' of', ' â\x80\x98', ' boxing', ',', ' which', ' is', ' itself', ' only', ' ever', ' present', ' in', ' any', ' individual', ' incarna
- **['negotiate']** — ['a', ' much', ' more', ' flexible', ' system', ' which', ' would', ' allow', ' local', ' managers', ' to', ' negotiate', ' local', ' rates', ' â\x80\x94', ' on', ' a', ' national', ' pay', ' spine', ' â\x80\x94', ' is', ' needed', '.', '\n']
- **['getting']** — ['lockean', ' children', ' have', ' the', ' added', ' amazing', ' ability', ' to', ' recognise', ' reason', ' in', ' you', ' without', ' possessing', ' any', ' themselves', ',', ' while', ' kantian', ' kids', ',', ' two', ' stages', ' back', ' from', ' rationality', ',', ' have', ' the', ' disconcer
- **['with']** — ['studies', ' in', ' this', ' area', ' have', ' indicated', ' that', ' too', ' strict', ' or', ' too', ' lax', ' parental', ' attitudes', ' increase', ' the', ' risk', ' of', ' delinquency', '(', ' the', ' work', ' of', ' john', ' bowlby', ',', ' for', ' example', ')', ';', ' they', ' have', ' found
- **['in']** — ['take', ' up', ' references', ' before', ' giving', ' credit', ',', ' lending', ' money', ',', ' letting', ' in', ' tenants', ',', ' taking', ' on', ' staff', '.', '\n']
- **['owed']** — ['as', ' long', ' as', ' you', ' can', ' show', ' that', ' they', ' owed', ' you', ' a', ' legal', ' duty', ' to', ' be', ' careful', ' and', ' have', ' been', ' negligent', ',', ' damages', ' could', ' come', ' your', ' way', '.', '\n']
- **['up']** — ['our', ' latest', ' foreign', ' secretary', ' has', ' yet', ' to', ' publicly', ' pick', ' up', ' the', ' file', ' marked', ' hong', ' kong', '.', '\n']
- **['after']** — ['and', ' also', ',', ' it', "'s", ' cos', ' that', ',', ' you', ' know', ' after', ' we', ' had', ' that', ' row', ' about', ' terry', ',', ' after', ' bank', ' holiday', ' expecting', ' not', ' to', ' have', ' to', ' do', ' a', ' day', ' in', ' the', ' shop', ',', ' and', ' er', ' he', "'s", ' alr

## 15 False Negatives (missed metaphors)

- **['to']** — ['pcbs', ' are', ' so', ' difficult', ' to', ' destroy', ',', ' that', ' rechem', "'s", ' emission', '-', 'monitoring', ' systems', ' are', ' geared', ' to', ' detecting', ' them', ' on', ' the', ' grounds', ' that', ' if', ' you', ' destroy', ' pcbs', ' you', ' destroy', ' everything', '.', '\n']
- **['over', '.']** — ['by', ' the', ' time', ' i', "'ve", ' paid', ' for', ' my', ' room', ' at', ' the', ' ywca', ' and', ' bought', ' all', ' the', ' make', '-', 'up', ' and', ' clothes', ' i', ' need', ' there', ' never', ' seems', ' to', ' be', ' anything', ' left', ' over', '.', '\n']
- **['taken']** — ['â\x80\x98', ' was', ' it', ' taken', ' down', ' by', ' a', ' policeman', '?', '\n']
- **['fleet']** — ['the', ' supertram', ' is', ',', ' in', ' effect', ',', ' a', ' low', '-', 'cost', ' replacement', ' for', ' british', ' rail', "'s", ' ageing', ' fleet', ' of', ' suburban', ' diesel', ' multiple', '-', 'unit', ' trains', ',', ' those', ' rasping', ' diesels', ' which', ' change', ' gear', ' to', 
- **['squad']** — ['and', ' if', ' he', ' does', "n't", ' do', ' that', ',', ' then', ' the', ' man', ' will', ' send', ' rickie', "'s", ' plump', ' young', ' ass', ' down', ' to', ' the', ' gang', '-', 'rape', ' squad', ' in', ' the', ' county', ' jail', '.', '\n']
- **['odd']** — ['â\x80\x98', ' it', ' may', ' seem', ' a', ' little', ' odd', ' for', ' prison', ' staff', ' to', ' be', ' looking', ' at', ' their', ' prisoners', ' as', ' customers', ',', ' but', ' they', ' are', ' the', ' direct', ' consumers', ' of', ' our', ' service', ',', ' explained', ' mr', ' stephen', ' 
- **['pupils', ',']** — ['first', ',', ' the', ' pupils', ',', ' teachers', ' and', ' practitioners', ' of', ' science', ' are', ' overwhelmingly', ' male', ' in', ' terms', ' of', ' numbers', ';', ' second', ',', ' school', ' science', ' is', ' packaged', ' and', ' presented', ' to', ' appeal', ' to', ' boys', ',', ' not'
- **['contrast', ',']** — ['in', ' contrast', ',', ' only', ' a', ' minority', ' of', ' girls', ' express', ' an', ' interest', ' in', ' science', ' at', ' any', ' age', ' but', ' there', ' is', ' no', ' obvious', ' swing', ' from', ' science', ' in', ' the', ' middle', ' and', ' late', ' teens', '.', '\n']
- **['interests']** — ['this', ' told', ' us', ' that', ' any', ' simplistic', ' assumptions', ' about', ' featuring', ' â\x80\x98', ' female', ' domestic', ' interests', ' in', ' science', ' would', ' probably', ' meet', ' with', ' failure', '.', '\n']
- **['â\x80\x98']** — ['â\x80\x98', ' she', ' allowed', ' a', ' man', ' to', ' persuade', ' her', ' into', ' taking', ' the', ' drug', ',', ' so', ' she', ' can', ' only', ' blame', ' herself', ' for', ' her', ' predicament', '.', '\n']
- **['spread']** — ['if', ' i', ' want', ' it', ' spread', ' around', '.', '\n']
- **['stretching']** — ['these', ' pipes', ' stretching', ' thinly', ' in', ' a', ' bridge', ' across', ' the', ' service', ' road', ' injected', ' a', ' mix', ' of', ' chemicals', ' into', ' the', ' walls', ' of', ' the', ' furnace', ' itself', ';', ' did', ' i', ' know', ' that', ' fuel', ' was', ' only', ' used', ' in'
- **['bow', ',']** — ['her', ' long', ' hair', ',', ' shining', ' gold', ',', ' was', ' swept', ' back', ' and', ' caught', ' at', ' the', ' nape', ' of', ' her', ' neck', ' with', ' a', ' bow', ',', ' make', '-', 'up', ',', ' expertly', ' applied', ',', ' accentuated', ' the', ' classically', ' beautiful', ' lines', ' 
- **['long']** — ['â\x80\x98', ' how', ' long', ' will', ' it', ' take', ',', ' anyway', '?', '\n']
- **['turn-of-the-century', ':']** — ['if', ' you', ' did', "n't", ' know', ' the', ' tunes', ',', ' you', "'d", ' assume', ' holly', "'s", ' lyrics', ' decorated', ' some', ' innocent', ' backporch', ' spooners', ' from', ' the', ' turn', '-', 'of', '-', 'the', '-', 'century', ':', ' â\x80\x98', ' stars', ' appear', ' and', ' the', ' 

## Prediction-file validation (Bạn 4 duty)
- 43,947 prediction rows == 43,947 test entries; idx contiguous/unique 0..43946
- Full row alignment verified: target token of each row corresponds to the same test entry
- **Normalization needed before shared scoring** (found 1,161 + 795 apparent diffs that are only formatting):
  - JSON targets keep original case/punctuation (`Design:`, `track,`); loader output lowercases and splits them
  - Possessives split differently: `London's` (JSON) vs `['london', "'s"]` (loader)
  - Hyphenated compounds split by the loader: `single-car` vs `single - car`
  - Rule for the shared script: lowercase; strip; split on `-` and punctuation; rejoin possessives; compare token spans, not raw strings
- Most frequent FP/FN targets are function words (on, that, in, to, with, at, up, get, have) - same POS classes that make VUA hard for all systems
