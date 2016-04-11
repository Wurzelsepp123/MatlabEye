function [ D ] = classify_data(data,mean_class,cov, prior, transmat, mu, Sigma, mixmat, option )
% classifies the data using the HMM defined by prior, transmat, mu Sigma
% and mixmat and calculates the distances defined by mean_class and cov.
% option determines the classifier-type ('eu' for euclidian, 'ma' for
% mahalanobis)

% data is the data to be tested

% In D each column corresponds to a sequence and every but the last row
% corresponds the distance to each mean of one class using the classifier
% defined with 'option'.
% In the last row the row with the smallest distance (classified type) is
% indicated

if ~iscell(data)
    data2{1}=data;
else
    data2=data;
end

[ ~ , vectors]=generate_classifier( data2, prior, transmat, mu, Sigma, mixmat, option );
for j=1:length(prior)
    if option=='eu'
        W{j}=eye(length(prior));
    elseif option=='ma'
        W{j}=inv(cov{j});
    end
end

for i = 1:size(vectors,2)
    for j=1:length(prior)              
        D(j,i)=(vectors(:,i)-mean_class{j})'*W{j}*(vectors(:,i)-mean_class{j});
    end
     D(length(prior)+1,i)=find(D(1:length(prior),i)==min(D(1:length(prior),i)));
end


end

