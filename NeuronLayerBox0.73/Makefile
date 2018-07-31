test : SM1-0.cu
	gcc -D_REENTRANT qt-show.c -o show -lpthread -lglut -lGLU -lGL
	nvcc -o test SM1-0.cu 
clean:  
	rm -rf *.o Test  
	rm -rf test
	rm -rf show
	@echo "Clean done!"  
