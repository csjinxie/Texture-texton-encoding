function GenMR8
% Generate MR filter response  of texture image
% MR8 reference: A statistical approach to texture classification from
% single images, IJCV, Manik Varma and Andrew Zisserman.
%If you generate texture features in other texture datasets, please modify
%rootpic and DataName.
%If there are some bugs or problems, please contact csjxie@gmail.com

rootpic = 'E:\texture_database\curetgrey\';
DataName = 'E:\MR8_CUReT'; % the image is normalized to 0 mean, 1 standard

classNum = 61;
picCount = 0;
NORIENT=6;
MR8Filters = makeRFSfilters(NORIENT);

for i=1:classNum
    for j=1:190
        filename = sprintf('%ssample%02d\\%02d-%03d.png', rootpic, i,i,j);
        fid = fopen(filename);
        if fid~=-1
            picCount = picCount+1;
            fclose(fid);
            Gray = imread(filename);
            Gray = im2double(Gray);
            Gray_norm = (Gray-mean(Gray(:)))/std(Gray(:));
            for j=1:size(MR8Filters,3)
                Temp = conv2(Gray_norm,MR8Filters(:,:,j),'valid');
                Temp = reshape(Temp,prod(size(Temp)),1);
                Response(:,j) = Temp;
            end

            for j=1:6 
                TempResponse = Response(:,(j-1)*NORIENT+1:j*NORIENT);
                MRTemp(:,j) = max(TempResponse,[],2);
            end

            MRTemp(:,7:8) = Response(:,6*NORIENT+1:NORIENT*6+2);
            % MRTemp Normalization as showed in Paper
            MRTemp2 = MRTemp.^2;
            MRMag = sqrt(sum(MRTemp2,2));
            NormFactor = log(1+MRMag/0.03)./MRMag;
            NormFactorExtend = repmat(NormFactor,1,8);    
            MR8Norm = MRTemp.*NormFactorExtend;
            matfile = sprintf('%s\\%04d',DataName,picCount); 
            save(matfile,'MR8Norm')
        end                
    end
end