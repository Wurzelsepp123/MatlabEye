function [ mean_v , vectors,  cov ] = generate_classifier( data, prior, transmat, mu, Sigma, mixmat, option )
% generates distance classifier 
% for mahalanobis set option = 'ma' / for euklid set option = 'eu'

% prior, transmat, mu, Sigma, mixmat are the trained HMM-parameter with
% each index is used for a different HMM (SPU/SPD/FIX)

% the output vectors contains the values for each sequences returned from
% each HMM (used in classify_data())

vectors=zeros(length(prior),length(data));

for i=1:length(data)
    for j=1:length(prior)
        vectors(j,i) = mhmm_logprob(data{i}, prior{j}, transmat{j}, mu{j}, Sigma{j}, mixmat{j});
        if vectors(j,i)==-Inf
           vectors(j,i)=-100000; 
        end
    end
end
mean_v = mean(vectors,2);
if option=='eu'
    cov=0;
elseif option=='ma'
    
    cov=zeros(length(prior));
    for i=1:length(data)
        cov=cov+vectors(:,i)*vectors(:,i)'-mean_v*mean_v';
    end
    cov=cov/length(data);
end
        
end

