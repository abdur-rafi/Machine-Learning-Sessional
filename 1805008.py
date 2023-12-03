import pandas as pd

class TelcoCustomer:

    def __init__(self, filePath) -> None:
        self.filePath = filePath

    def load(self):
        self.df = pd.read_csv(self.filePath)
        return self
    
    def summarize(self):
        print(self.df.describe())
        return self

    def print(self):
        print(self.df)
        print(self.df.value_counts())
        return self

def main():
    tc = TelcoCustomer("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # tc.load().print().summarize()
    tc.load()

    # print(tc.df.head(5))
    # print(tc.df.info())
    print(tc.df.value_counts())



if __name__ == "__main__":
    main()

print("hello world")
