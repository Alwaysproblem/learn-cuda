all: test_section

test_section: my_section_data.o main.o
	clang++ my_section_data.o main.o -o test_section -Wl,-T,myld.ld

my_section_data.o: my_section_data.cpp
	clang++ -c my_section_data.cpp

main.o: main.cpp
	clang++ -c main.cpp

clean:
	rm -f *.o test_section
