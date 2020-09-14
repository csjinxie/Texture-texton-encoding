function  [Histogram_c, Histogram_r]=TEISF(Texton, Texture_feature, P, S, lambda)
%  TEISF:Texton induced statistical features for texture classification
%  [Histogram_c, Histogram_r]=TEISF(Texton, Texture_feature, P, S) returns the coefficient histogram and residual histogram 
%  Input: Texton is the learned dictionary 
% Texture_feature is the texture feature such as MR8 or Patch
% P is the number of atoms in the formed sub-dictionary
%  S is the sample rate 
% lambda: regularization parameter
% Output:Histogram_c and Histogram_r are the coefficient and residual histograms
% Authors: Jin Xie, Lei Zhang and Jane You
%If there are some bugs or problems, please contact csjxie@gmail.com
% Copyright @ Biometrics Research Centre, the Hong Kong Polytechnic University


if nargin<5
    disp('Not enough input parameters.')
    return
end

[n,k]=size(Texton);
% P=100;
% S=3;
% lambda=1.5;
Histogram_c=zeros(1,k);
Histogram_r=zeros(1,k);
for f=1:size(Texture_feature,1)
    Coefficient_c=zeros(1,k);
    Coefficient_r=zeros(1,k);
    Texture_feature(f,:)=Texture_feature(f,:)./norm(Texture_feature(f,:));    %normalization
     mm=Texton'*Texture_feature(f,:)';
     %xx=repmat(mm',size(Texton,1),1);
     yy=Texton.*repmat(mm',size(Texton,1),1);
     proj=sum(((repmat(Texture_feature(f,:),size(Texton,2),1))'-yy).^2,1); % compute the distance between the texture feature and the atoms in the dictionary
    [va,in]=sort(sqrt(proj'));
     Coe=inv(Texton(:,in(1:S:S*P))'*Texton(:,in(1:S:S*P))+lambda*eye(P))*Texton(:,in(1:S:S*P))'*Texture_feature(f,:)'; % the encoding coefficient
     %bb=Texton(:,in(1:S:S*P)).*repmat(Coe',n,1); 
     Dis=sum((Texton(:,in(1:S:S*P)).*repmat(Coe',n,1)-repmat(Texture_feature(f,:),P,1)').^2,1); %the encoding residual
      tt=ones(P,1)./sqrt(Dis');
    Coefficient_r(in(1:S:S*P))=tt./sum(tt);
    Coefficient_c(in(1:S:S*P))=abs(Coe')./sum(abs(Coe));
    Histogram_c=Histogram_c+Coefficient_c;
    Histogram_r=Histogram_r+Coefficient_r;
end
  
   