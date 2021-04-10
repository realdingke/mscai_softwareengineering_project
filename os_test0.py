import os

if __name__ == "__main__":
    print(os.path.dirname(os.path.abspath(__file__)))   #os.path.dirname(__file__) requires full file path name
    print(os.getcwd())