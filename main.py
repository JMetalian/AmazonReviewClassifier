from SVCMethod import LinearSupportVectorClassifier
from DecisionTreeMethod import DesTree


method = input("Enter '1' for SVC\n"
                "Enter '2' for Decision Tree\n"
                
               "\n:").strip().strip("'").strip('"').lower()
load = input("\nLoad already trained model?\n"
             "Enter 'Y' for previous model.\n"
             "Enter 'N' for a new model training."
             "\n:").strip().strip("'").strip('"').lower()

load = load != "N"
if method == "1":
    LinearSupportVectorClassifier(load)
##YOUR CODE###
elif method == "2":
    DesTree(load)


##############
else:
    print("Invalid Input")