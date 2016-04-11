function [ data ] = load_data( filepath )
% loads .coord file into array with size(data)=[#samples 4]

data=textread(filepath,'%f','delimiter',' ','headerlines',11);
data=reshape(data,4,size(data,1)/4)';

end

