tic
%%
% This is the driver function for all the function implemented in this lab.
% change the parameters and call the corresponding functions.


%%
[X,Y]=inputImg;   % image matrix and corresponding label
concatedMatrix=[X Y];  %21999x1025
s=size(X,1);
rmp=randperm(s);
a=int16(0.8*s);  % partitioning the dataset into 80:20 ratio
TrainMat=(concatedMatrix(rmp(1:a),:));  % Shuffling the dataset 
TestMat=(concatedMatrix(rmp(a+1:end),:));

TrainX=TrainMat(:,1:end-1);  % now separating the Image data and corresponding steering angle
TrainY=TrainMat(:,end);

TestX=TestMat(:,1:end-1);
TestY=TestMat(:,end);   

nEpochs=1000;
batch_size=64;
H1=512;
H2=64;
eta=0.01;

%% here testerror is the validation error plotted in the graphs
[u,v,w,trainerror,testerror]=Question2(TrainX,TrainY,TestX,TestY,eta,H1,H2,nEpochs,batch_size); % comment this function call if you want call dropout function

%[u,v,w,trainerror,testerror]=mlDropout(TrainX,TrainY,TestX,TestY,eta,H1,H2,nEpochs,batch_size);
%uncomment this dropout if you have to call dropout function.

save('Param_values.mat','u','v','w','trainerror','testerror');  % we can change it according to our need

dd=linspace(1,nEpochs,nEpochs);
 plot(dd,trainerror,'-r','LineWidth',2);
  xlabel('epochs');
  ylabel('error');
  title('batch size: 64  learning rate : 0.01');
     hold on;
  plot(dd,testerror,'-b','LineWidth',2);
legend('training set','Validation Set');

 hold off;

toc






