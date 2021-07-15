# FixedPointNN
to compile: run command 
go build
in the directory containing the code files

to run: run command
./fixedpointnn -flag args
followed by flags

flags:
mnist (train or predict with mnist numbers dataset)
file (run prediction on specific image file

args:
  mnist
  -train (trains net)
  -val (generates validation set for model comparison)
  -plot (trains and validates multiple iterations of model to test for accuracy at varying weight ranges)
  -predict (shows accuracy of stored model)
  
  file
  -FILENAME (local address of file to have prediction run on)
  
