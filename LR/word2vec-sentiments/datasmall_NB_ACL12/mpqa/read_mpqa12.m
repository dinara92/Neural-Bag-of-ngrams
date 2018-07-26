%%
useTrees = 0;

file1 = './mpqa_t4.dat';
fid1 = fopen(file1);

%%
sentence_words = {};
training_labels = [];
testing_labels = [];

tic
labels = zeros(1,10^5);
sentence_words = cell(1,10^5);
counter = 0;
while ~feof(fid1)
        try
        tempstr = fgetl(fid1);
        while strcmp(tempstr(end),' ')||strcmp(tempstr(end),'.')
           tempstr(end) = [];
        end
        counter = counter + 1;
        temp_split = regexp(tempstr,'\t','split');
        labels(counter) = str2num(temp_split{1});
        sentence_words{counter} = regexp(temp_split{2},' ','split');
        catch
            keyboard
        end
end

sentence_words(counter+1:end) = [];
labels(counter+1:end) = [];
toc

num_examples = counter;
%%

%file = strcat('../data/embeddings-scaled.',num2str(embedding_size),'.mat');
%load(file);

wordMap = containers.Map(words,1:length(words));

wordMap('elipsiseelliippssiiss') = wordMap('...');     
wordMap('smilessmmiillee') = (uint32(wordMap.Count) + 1);   %end-4
% We = [We We(:,wordMap('smile'))];
wordMap('frownffrroowwnn') = (uint32(wordMap.Count) + 1);   %end-3
% We = [We We(:,wordMap('frown'))];
wordMap('haha') = (uint32(wordMap.Count) + 1);      %end-2
% We = [We We(:,wordMap('laugh'))];
wordMap('hahaha') = (uint32(wordMap.Count));        %end-2
% We = [We We(:,wordMap('laugh'))];
wordMap('hahahaha') = (uint32(wordMap.Count));      %end-2
% We = [We We(:,wordMap('laugh'))];
wordMap('hahahahaha') = (uint32(wordMap.Count));    %end-2
% We = [We We(:,wordMap('laugh'))];
wordMap('hehe') = (uint32(wordMap.Count) + 1);      %end-1
% We = [We We(:,wordMap('laugh'))];
wordMap('hehehe') = (uint32(wordMap.Count));        %end-1
% We = [We We(:,wordMap('laugh'))];
wordMap('hehehehe') = (uint32(wordMap.Count));      %end-1
% We = [We We(:,wordMap('laugh'))];
wordMap('hehehehehe') = (uint32(wordMap.Count));    %end-1
% We = [We We(:,wordMap('laugh'))];
wordMap('lol') = (uint32(wordMap.Count) + 1);       %end
% We = [We We(:,wordMap('laugh'))];
wordMap('lolol') = (uint32(wordMap.Count));         %end
% We = [We We(:,wordMap('laugh'))];
words = [words {'elipsiseelliippssiiss'} {'smilessmmiillee'} {'frownffrroowwnn'} {'haha'} {'hahaha'} {'hahahaha'} {'hahahahaha'} {'hehe'} {'hehehe'} {'hehehehe'} {'hehehehehe'} {'lol'} {'lolol'}];
%num_examples = 1000;
words_indexed = cell(num_examples,1);
words_reIndexed = cell(num_examples,1);
%words_reIndexed2 = cell(num_examples(2),1);
words_embedded = cell(num_examples,1);
sentence_length = cell(num_examples,1);
index_list = [];

tic
unknown_words = [];
for i=1:num_examples
    if mod(i,1000)==0
        disp([num2str(i) '/' num2str(num_examples)]);
    end
    
   % if useTrees
  %      words_indexed{i} = allSNum{i};    
  %  else
        words_indexed{i} = cellfun(@(x) wordMap(x),sentence_words{i});
  %  end
    
  if(words_indexed{i} == 1)
      unknown_words = [unknown_words sentence_words{i}];
  end
    
%     words_embedded{i} = We(:,words_indexed{i});
    sentence_length{i} = length(words_indexed{i});
    
end
toc
%%
index_list = cell2mat(words_indexed');
unq = sort(index_list);
freq = histc(index_list,unq);
unq(freq==0) = [];
freq(freq==0) = [];

reIndexMap = containers.Map(unq,1:length(unq));
words2 = words(unq);

parfor i=1:num_examples
    words_reIndexed{i} = arrayfun(@(x) reIndexMap(x), words_indexed{i});
end

% We2 = We(:, unq);
%%

% cv_obj = cvpartition(labels,'kfold',10);
% save('cv_obj_mpqa','cv_obj');

load('cv_obj_mpqa');

%%
%CVNUM = 10;
full_train_ind = cv_obj.training(CVNUM);
full_train_nums = find(full_train_ind);
test_ind = cv_obj.test(CVNUM);
test_nums = find(test_ind);

train_ind = full_train_ind;
cv_ind = test_ind;

% full_train_nums = randsample(1:num_examples,floor(num_examples*0.9));
% full_train_ind = ismember(1:num_examples,full_train_nums);
% test_nums = 1:num_examples;
% test_nums(full_train_nums)=[];
% 
% temp_ind = randsample(1:length(full_train_nums),floor(length(full_train_nums)*0.7));
% train_nums = full_train_nums(temp_ind);
% cv_nums = full_train_nums;
% cv_nums(temp_ind) = [];
% 
% train_ind = ismember(1:num_examples, train_nums);
% cv_ind = ismember(1:num_examples, cv_nums);
% 
% test_ind = ismember(1:num_examples, test_nums);

%save('EP_rand_set', 'full_train_ind', 'full_train_nums', 'test_nums' , 'temp_ind', 'train_nums','cv_nums','train_ind','cv_ind','test_ind');
%load('EP_rand_set.mat');
allSNum = words_reIndexed;
if ~useTrees
    allSKids = {};
    allSTree = {};
end

clear sentence_words_temp

isnonZero = ones(1,length(allSNum));


sent_freq = ones(length(allSNum),1);
