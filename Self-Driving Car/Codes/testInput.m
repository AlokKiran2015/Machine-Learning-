function [Y]=testInput()
myfile = fopen('test/test-data.txt','rt');
count = 0;

while (fgets(myfile) ~= -1),
  count = count+1;
end
file=importdata('test/test-data.txt');
Y=zeros(count,1024);

for i=1:count
    
   A=imread(strcat('test/',file{i}(8:end)));
   A=rgb2gray(A);
   
    A=double(A);
A=A/255;
Y(i,:)=reshape(A,1,[]);
end
end