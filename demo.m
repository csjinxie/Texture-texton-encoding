%Demo program of texture classification
%Use GenMR8 to generate MR8 features of texture images and save them in MR8Folder.
%Use kmeans clustering, L1 norm minimization or texton_learning to learn textons. We
%put the learned textons in the subfoldters.
%In the demo program, we randomly choose training samples to obtain
%classification accuracy. Please repeat many times to calculate the mean
%classification accuracy.
%%If there are some bugs or problems, please contact csjxie@gmail.com

 load '.\CUReT\23_trainingsamples\Texton';
 MR8Folder = '.\CUReT\MR8_CUReT';
 TrainNumPerClass = 23;
ClassNum = 61;
PicNumPerClass = 92;
Weight=0.55;
flag=0;
 for i=1:ClassNum
   for j=1:PicNumPerClass
     flag=flag+1;
    matfile = sprintf('%s\\%04d',MR8Folder, flag);
    load(matfile)
    Texture_feature=MR8Norm;
    [Histogram_c, Histogram_r]=TEISF(Texton, Texture_feature, 100, 3, 1.5);
    AllMRHists_c(:,flag)=Histogram_c';
    AllMRHists_r(:,flag)=Histogram_r';
     sprintf('samplenums=%d',flag)
   end
 end

 p = randperm(PicNumPerClass);
randTrain = p(1:TrainNumPerClass);
randTest = p(TrainNumPerClass+1:PicNumPerClass);
for i=1:ClassNum
    TrainIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = randTrain+(i-1)*PicNumPerClass;
   TrainClassIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = i;
    TestIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = randTest+(i-1)*PicNumPerClass;
   TestClassIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = i;
end

CP=Classify_NSC(AllMRHists_c, AllMRHists_r, TrainIDs,TestIDs,TestClassIDs, ClassNum,TrainNumPerClass, Weight);

 