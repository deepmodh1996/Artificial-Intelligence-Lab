	set term png
	set datafile separator ","


###############################################

	set ylabel "Accuracy in percentage  ----->"
	set xlabel "Size of Dataset ----->"



#######################################################

	set output "q3-accuracy-graph.png"



#################3##########################################

	#plot "q1-Confidence-factor.csv" u 1:2 title "Training Data" with linespoints, \
	      	"q1-Confidence-factor.csv" u 1:3 title "Test Data"  with linespoints



#############################################################


  #plot "q3-time.csv" u 1:2 title "J48" with linespoints, \
	      	"q3-time.csv" u 1:3 title "MultiLayer perceptron"  with linespoints




#######################################################

  plot "q3-accuracy.csv" u 1:2 title "J48" with linespoints, \
	      	"q3-accuracy.csv" u 1:3 title "MultiLayer perceptron"  with linespoints

