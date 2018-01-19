function [ out , Y] = inputImg()
tic
file = fopen('steering/data.txt','rt');
count = 0;
while (fgets(file) ~= -1),
  count = count+1;
end
myfile = importdata('steering/data.txt');
out=zeros(count,1024);
label=zeros(count,1);

for i=1:count
    A=imread(strcat('steering/',myfile.textdata{i}(3:end)));
label(i)=myfile.data(i);
A=rgb2gray(A);
A=double(A);
A=A/255;
out(i,:)=reshape(A,1,[]);
Y=label;
end
toc
end