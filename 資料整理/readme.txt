1. 彙整資料
    - 1.1 將重複N結果(附檔S1_H1N1_no_001_result_1.txt)使用S2_takedata.py將結果儲存成S3_H1N1_no_001_result.csv
    - 1.2 使用S4_trans.py轉置S3_H1N1_no_001_result.csv為S5_H1N1_no_001_result_trans.csv
    - P.S.如果批次黨能不寫成N個檔即不需要做1.1動作
    - P.S.2 S2_takedata.py可以更改寫法:藉由讀取所有行數並減去上面非對稱行數(顯示TP、TN等)並除二可得每次資料個數(程式內為手動輸入)
    - P.S.3 S4_trans.py可以更改寫法:寫入檔案時除了第一次的label外不寫入其他次的label(S5_H1N1_no_001_result_trans.csv為手動刪除除第一行外的其他24次label)
    - P.S.4 S1_H1N1_no_001_result_1.txt裡有tp,tn,fp,fn可以直接計算precision,sensitivity,specificity,mcc，公式可參考維基百科，計算結果如附檔S6_each_result.xlsx

2. 製作auroc、auprc
    - 2.1 將所有leave one out經過上述處理後結尾為"trans.csv"的檔做完25次平均後合併成一個檔案如附檔S7_H1N1_each.xlxs
    - 2.2 將S7_H1N1_each.xlxs放入

P.S.提供數據的大小供參考
S1:(4+1+176*2)行，所以行數可藉由讀去總行數後減5並除2得數據長度
S3:[2,176*25]，其中25為上述的N
S5:[2*25,176]，其中25為上述的N
S6:3種病毒(H1N1,H3N2,SARS)*5個數值(accuracy,precision,sensitivity,specificity,mcc)
S7:為2*SUM(leave1~24的行數，如上的leave1為176行，每個受試者的行數不一定相同)

P.S.提供檔案部分僅供參考，非同次製作，所以檔案大小、數值不同屬正常



 	