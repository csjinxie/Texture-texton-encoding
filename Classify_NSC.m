
function CP=Classify_NSC(DM_c, DM_r, TrainIDs,TestIDs,TestClassIDs, ClassNum,TrainNumPerClass, Weight)
%  Classify_NSC computes the classification accuracy with the fused nearest subspace classifier (NSC)
%  CP=ClassifyOnNN(DM_c, DM_r,trainClassIDs,testClassIDs) returns the classification accuracy 
%  Input: DM_c is a m*n coefficient feature matrix, m is the dimension of the coefficient feature, n is the number of the samples
% DM_r is a m*n  residual feature matrix, m is the dimension of the residual feature, n is the number of the samples (They are arranged class by class)
%  TrainIDs and TestIDs are the IDs of training and test samples (They are arranged class by class)
%  TestClassIDs is the class ID of the test samples
% ClassNum is the number of texture classes
%  TrainNumPerClass is the number of training samples per class
% Weight is the weight between the two NSC classifiers 
% Output:CP is the classification accuracy
% Authors: Jin Xie, Lei Zhang and Jane You
%If there are some bugs or problems, please contact csjxie@gmail.com
% Copyright @ Biometrics Research Centre, the Hong Kong Polytechnic University

if nargin<8
    disp('Not enough input parameters.')
    return
end

TrainMRHists_c=DM_c(:,TrainIDs);
TestMRHists_c=DM_c(:,TestIDs);
TrainMRHists_r=DM_r(:,TrainIDs);
TestMRHists_r=DM_r(:,TestIDs);

%the nearest subspace classifier
for j=1:ClassNum
    S_c=pinv(TrainMRHists_c(:,(j-1)*TrainNumPerClass+1:j*TrainNumPerClass))*TestMRHists_c;
    Err_c(j,:)=sqrt(sum((TrainMRHists_c(:,(j-1)*TrainNumPerClass+1:j*TrainNumPerClass)*S_c-TestMRHists_c).^2,1));
end

for j=1:ClassNum
    S_r=pinv(TrainMRHists_r(:,(j-1)*TrainNumPerClass+1:j*TrainNumPerClass))*TestMRHists_r;
    Err_r(j,:)=sqrt(sum((TrainMRHists_r(:,(j-1)*TrainNumPerClass+1:j*TrainNumPerClass)*S_r-TestMRHists_r).^2,1));
end

%the fusion of two classifiers
Err=Weight*Err_c+(1-Weight)* Err_r;
 [Distance, Index]= min(Err);
 rightCount =0;
for i=1:length(TestIDs)
 if Index(i) == TestClassIDs(i)
        rightCount = rightCount+1;
 end
end
CP = rightCount/length(TestClassIDs);

