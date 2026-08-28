import CNN
import transfer_learning_shallow as tl
import simple_perceptron as sp

goodinput = 0

while True:
    print("Which model to run? Enter a nummber:")
    ans = int(input("Pereptron: 1  CNN: 2  Tranfer Learning: 3 \n"))
    
    if ans in [1,2,3]:
      break 
  
match ans:
    case ans if ans == 1:
        sp.run()
    case ans if ans == 2:
        CNN.run()
    case ans if ans == 3:
        tl.run()

