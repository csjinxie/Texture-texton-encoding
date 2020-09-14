function ttexton=texton_learning(TrainNumPerClass, ClassNum, PicNumPerClass, MR8Folder, nt, S, P)
% An example of texton learning. The textons are learned class by class. You
% can also solve representation coefficient by L_1 norm minimization or the method in the paper.
%D: sub-dictionary per class
%ttexton:the concatenated dictionary class by class
%alpha:representation coefficient per class
%email:csjxie@gmail.com


TrainNumPerClass = 6;
strTrainMethod = 'alternative';
MR8Folder='E:\texture data\MR8_CUReT';
ClassNum = 61;
PicNumPerClass = 92;
switch strTrainMethod
    case 'alternative' % alternative
        % Modified set up training sets
        Step = floor(PicNumPerClass/TrainNumPerClass);
        for i=1:ClassNum
            trainIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = (i-1)*PicNumPerClass+1:Step:(i-1)*PicNumPerClass+Step*(TrainNumPerClass-1)+1;
            trainClassIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = i;
            T = [];
            for j=1:TrainNumPerClass-1
                T = [T,(i-1)*PicNumPerClass+(j-1)*Step+2:(i-1)*PicNumPerClass+j*Step];
            end
            T=[T,(i-1)*PicNumPerClass+Step*(TrainNumPerClass-1)+2:i*PicNumPerClass];
            testIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = T;
            testClassIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = i;
        end
    case 'first' % first
        % old method to set up training set
        for i=1:ClassNum
            trainIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = (i-1)*PicNumPerClass+1:(i-1)*PicNumPerClass+TrainNumPerClass;
            trainClassIDs((i-1)*TrainNumPerClass+1:i*TrainNumPerClass) = i;
            testIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = (i-1)*PicNumPerClass+TrainNumPerClass+1:i*PicNumPerClass;
            testClassIDs((i-1)*(PicNumPerClass-TrainNumPerClass)+1:i*(PicNumPerClass-TrainNumPerClass)) = i;
        end
end

%Texton learning
DataName='E:\CUReT_Texture data';
ttexton=[];
nt=100; % texton number per class
 P=21;
 S=3;
for k=1:ClassNum
%tic 
 index = find(trainClassIDs==k);
    MR8OneClass = [];
    for j=1:length(index)
        matfile = sprintf('%s/%04d',MR8Folder,trainIDs(index(j)));
        load(matfile)
         MR8Norm = MR8Norm(1:2:end,:); % to reduce feature size, otherwise it will be out of memory
        MR8OneClass = [MR8OneClass;MR8Norm];
    end
    D=MR8OneClass(1:2:2*nt,:)';  % inialization
    for i=1: nt            % normalization
    xn=sum(D(:,i).^2);
    D(:,i) = D(:,i)./sqrt(xn);
    end     
 J = 100;
 iteration=1;
while (iteration<30)
    % solve alpha
    alpha=zeros(nt,size(MR8OneClass,1));
     residual=MR8OneClass';
    for f=1:size(MR8OneClass,1)
        mm=D'*residual(:,f);
        xx=repmat(mm',size(D,1),1);
        yy=D.*xx;
        proj=sum(((repmat(residual(:,f)',size(D,2),1))'-yy).^2,1);
        [val,index]=sort(abs(proj'));
        aa=pinv(D(:,index(1:S:S*P)))*residual(:,f);  % number of the selected atoms should be smaller than the dimension of MR8 feature, otherwise add the regularization term
        residual(:,f)=residual(:,f)-D(:,index(1:S:S*P))*aa;
        alpha(index(1:S:S*P),f)=aa;   
    end
    
    
%        sum_alpha=inv(D'*D+lamda*I)*D'*sum(residual,2);    % solve alpha with the regularized least square problem (by adding the l_2 norm regualrization term and mean subtraction term) 
%       for f=1:size(MR8OneClass,1)
%           alpha(:,f)=inv(D'*D+lamda*I+garma*I)*(D'*residual(:,f)+garma*sum_alpha/nt);
%       end
    
    
   Old_J = J;
   J=norm(residual);
   if abs(J-Old_J)<0.005    % the second stopping rule
       upd = false;
   end
    % update D
    for i=1:nt  
        %Y = zeros(8,size(MR8OneClass,1));
        D(:,i)=zeros(8,1);
        alpha1=alpha(i,:);
        alpha(i,:)=zeros(1,size(alpha,2));
        Y=D*alpha;
        Y = MR8OneClass' - Y;
        di = Y*alpha1'/sqrt(norm(Y*alpha1'));
        New_D(:,i) = di;   
    end
    % normalization
    for i=1: nt 
    xn2=sum(New_D(:,i).^2);
    New_D(:,i) = New_D(:,i)./sqrt(xn2);
    end    
    D = New_D; 
    iteration=iteration+1;
    sprintf('iteration number=%d',iteration)
end
%toc
sprintf('texton class number=%d',k)
ttexton=[ttexton D];
end
matfile = sprintf('%s\\%s\\textons',DataName,num2str(nt)); 
save(matfile,'ttexton')