程式碼主要函式：
def PreprocessData(raw_df):要預處理的資料(raw_df)作為輸入，分別回傳feature、label

def create_pairs(x, train_symptoms):製作訓練資料，將預處理好的資料作為輸入(x代表features、train_symptoms代表基礎label)，回傳配對完成並且將順序打亂的pair的feature、label

def create_test_pairs(x, train_symptoms, y, test_symptoms):製作validaition資料，將預處理好的資料作為輸入(x代表訓練資料features、train_symptoms代表訓練資料的基礎label
							   y:validation資料的features、test_symptoms:validation資料的基礎label)，回傳配對完成的驗證資料的feature、label

def create_standar_test_pairs(x, standar_symptoms, y, test_symptoms): 製作leave one out資料進行測試，將預處理好的資料作為輸入(x:代表健康樣本的feature、standar_symptoms
							   代表健康樣本的基礎label、y:leave one out資料的features,test_symptoms:leave one out資料的 basic label)，回傳
							   配對完成的驗證資料的feature、label

def create_base_net(input_shape):製作孿生網路，孿生神經網路結構由此修改

def show_train_history(train_history,train,test):顯示每個epoch的訓練結果，以圖片的方式呈現






