
%load('results.mat')
precision = 10;
filename = 'Classifier_Data2.cs';

format long

sequencelength = result{end,3};
thresh_spd = 0.4;
thresh_spu = 0.6;
trans = result{end,12};
mu = result{end,13};
Sigma = result{end,14};
prior = result{end,11};
mean_class = result{end,16};
cov = result{end,17};
filter_coeff = result{end,5};

names={'spu','spd','fix'};

%%
%%%%%%%%%%%
% Header  %
%%%%%%%%%%%

fileID = fopen(filename,'w');
fprintf(fileID,[...
    'using System;\n'...
    'using System.Collections.Generic;\n'...
    'using System.Linq;\n'...
    'using System.Text;\n'...
    'using System.Threading.Tasks;\n'...
    'using Accord.Audio;\n'...
    'using Accord.Audio.Filters;\n'...
    'using Accord.Statistics.Models.Markov;\n'...
    'using Accord.Statistics.Distributions.Univariate;\n'...
    '\n'...
    'namespace Classifier\n'...
    '{\n'...
    '\tpublic partial class SPOCKClassifier\n'...
    '\t{\n'...
    ]);   

%%
%%%%%%%%%%%%%%%%%%%
% Write Parameter %
%%%%%%%%%%%%%%%%%%%

fprintf(fileID,[...
    '\t\tconst int c_windowSize = ' num2str(sequencelength) ';\n'...
    '\n'...
    '\t\tconst double c_spu_threshold = ' num2str(thresh_spu) ';\n'...
    '\n'...
    '\t\tconst double c_spd_threshold = ' num2str(thresh_spd) ';\n'...
    '\n'...
    '\t\tconst int c_kernelSize = ' num2str(length(filter_coeff)) ';\n'...
    '\n'...
    ]);   
fprintf(fileID,['\t\treadonly double[] c_filter_kernel = new double[' num2str(length(filter_coeff)) '] { \n']);
for ind = 1:length(filter_coeff)
    fprintf(fileID,['\t\t\t' num2str(filter_coeff(ind),precision) ',\n']);
end
fprintf(fileID,'\t\t};\n\n');

%%
%%%%%%%%%%%%%%%%%%%%%%%
% Write HMM-Parameter %
%%%%%%%%%%%%%%%%%%%%%%%

fprintf(fileID,'\t\t//HMMs\n\n');
for hmms=1:length(names)
    fprintf(fileID,['\t\t\t//' names{hmms} '\n' ]);
    %% c_xxx_Trans
    fprintf(fileID,['\t\treadonly double[,] c_' names{hmms} '_Trans = new double[' num2str(size(trans{hmms},1)) ', ' num2str(size(trans{hmms},2)) '] {\n']);
    for row=1:size(trans{hmms},1)
        str = ['\t\t\t{ '];
        for col=1:size(trans{hmms},2)
            str = [ str num2str(trans{hmms}(row,col),precision) ', '];
        end
        str = [ str '},\n'];
        fprintf(fileID,str);
    end
    fprintf(fileID,'\t\t};\n'); 

    %% c_xxx_Distr
    fprintf(fileID,['\t\treadonly NormalDistribution[] c_' names{hmms} '_Distr = new NormalDistribution[' num2str(length(mu{hmms})) '] {\n']); 
    for ind=1:length(mu{2})
        fprintf(fileID,['\t\t\t new NormalDistribution( ' num2str(mu{hmms}(ind),precision) ', ' num2str(Sigma{hmms}(ind),precision) ' ),\n']);
    end
    fprintf(fileID,'\t\t};\n'); 
    %% c_spd_entryProb
    fprintf(fileID,['\t\treadonly double[] c_' names{hmms} '_entryProb = new double[' num2str(length(prior{2})) '] {\n']); 
    for ind=1:length(prior{hmms})
        fprintf(fileID,['\t\t\t' num2str(prior{hmms}(ind),precision) ',\n']);
    end
    fprintf(fileID,'\t\t};\n\n'); 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write Mahalanobis-Parameter %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(fileID,'\t\t//Mahalanobis\n\n');
for hmms=1:length(names)
    fprintf(fileID,['\t\t\t//' names{hmms} '\n' ]);
    %% xxx_midpoint
    fprintf(fileID,['\t\treadonly double[] ' names{hmms} '_midpoint = new double[' num2str(length(mean_class{hmms})) '] {\n']);
    for ind=1:length(mean_class{2})
        fprintf(fileID,['\t\t\t' num2str(mean_class{hmms}(ind),precision) ',\n']);
    end
    fprintf(fileID,'\t\t};\n'); 

    %% c_xxx_invCov
    fprintf(fileID,['\t\treadonly double[,] c_' names{hmms} '_invCov = new double[' num2str(size(cov{hmms},1)) ', ' num2str(size(cov{hmms},2)) '] {\n']); 
    for row=1:size(cov{hmms},1)
        str = ['\t\t\t{ '];
        for col=1:size(cov{hmms},2)
            str = [ str num2str(cov{hmms}(row,col),precision) ', '];
        end
        str = [ str '},\n'];
        fprintf(fileID,str);
    end
    fprintf(fileID,'\t\t};\n'); 
end
fprintf(fileID,'\t}\n}'); 
fclose(fileID);