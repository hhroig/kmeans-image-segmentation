root = cd;
%cd 'Training' % uncomment to segment and compare in Training set!
cd 'Test' % uncomment to segment and compare in Test set!
 
%% get filenames
names_pp = dir('*.ppm'); % real image
names_pb = dir('*.pbm'); % segmentation

%% initialize cells to save images!
segmented= cell(1, length(names_pb)); % expert segmentation
real = segmented; % dermatoscopes images
my_segmentation = segmented;  % proposed segmentation
pure_clustered = my_segmentation; % just to check pure cluster-segmentations...

%% initialize arrays to save Jaccard indexes!
similarity = zeros(1, length(names_pb));
pure_similarity = similarity;

for k=1:length(names_pb)
    tic
        %% loading expert segmentations
        Im_segmented = imread(names_pb(k).name);
        segmented{k} = Im_segmented; %Im_gray(:); % every entry is a face
        
        %% loading colored images from dermatoscopes:        
        real{k} = imread(names_pp(k).name);   % we save real image!
        
        %% now we do K-means to dermatoscope's image
        Im = double(real{k});
        [nrows, ncols, ~] = size(Im);
        
        temp = reshape(Im, [nrows*ncols,3]);
                
        % clustering each image after gray-scaled:
        [idx, ~] = kmeans(temp,2);  %,'Distance' , 'cityblock','MaxIter',1000); % 'Distance' 'sqeuclidean' 'cityblock'  'cosine'  'correlation'  'hamming'
        temp = reshape(idx, [nrows, ncols])- ones([nrows, ncols]);
        
        %% putting 1's for injury and 0's for healthy skin!
        row_core = 1:nrows;
        row_core(round(0.25*nrows):round(0.75*nrows)) = [];
        
        cols_core = 1:ncols;
        cols_core(round(0.25*ncols):round(0.75*ncols)) = [];
        core_temp = temp(row_core, cols_core);
        
        if sum(core_temp(:)) > (1/2)*length(row_core)*length(cols_core)
            clustered = logical(temp-1);
        else
            clustered = logical(temp);
        end
                
        pure_clustered{k} = clustered;
        %% Post-Processing!
        
        % Detecting the black frames!
        % just to check where are the "blacks" in the image!
         red = Im(:,:,1);
         green = Im(:,:,2);
         blue = Im(:,:,3);
         
         % remove the "blacks" by checking inside some of the 3 matrix colors:
         my2_seg = clustered;
         black_ind = logical( (red<20).*(blue<20).*(green<20) );
         my2_seg(black_ind) = 0;
                
         % then erode!
         SE1 = strel('disk',1);
         eroded = imerode(my2_seg, SE1);         
         
         % then fill holes
         filled = imfill(eroded, 'holes');         
        
         %then dilate:
         SE2 = strel('disk',8); 
         dilated = imdilate(filled,SE2);
         
        % then find connected components and erase the smaller regions of
        % 1's
         BW = dilated;
         CC = bwconncomp(BW);
         numPixels = cellfun(@numel,CC.PixelIdxList);
         idx = find(numPixels < quantile(numPixels, 0.99)); 
         for j = idx
         BW(CC.PixelIdxList{j}) = 0;
         end
         
         % save the final segmentation!         
         my_segmentation{k} = BW;
         
        %% Jaccard similarity Index
        similarity(k) = jaccard(my_segmentation{k}, segmented{k}); 
        pure_similarity(k) = jaccard(pure_clustered{k}, segmented{k});
      toc
end

cd(root)
%% Checking graphs...
for k=1:length(my_segmentation)
        figure
        subplot(2,2,[1,2]),imshow(real{k})
        title('Dermatoscope Image')
        subplot(2,2,3),imshow(segmented{k})
        title('Expert Segmentation')
        subplot(2,2,4),imshow(my_segmentation{k})
        title(['Proposed Segmentation. J = ', num2str(similarity(k))])
end

%% Projections:
for k=1:length(my_segmentation)
        figure
        imshowpair(my_segmentation{k}, segmented{k})
        title(['Jaccard Index = ' num2str(similarity(k))])
end 

%% Jaccard Index:
% let's check how good are my segmentations!
figure
histogram(similarity)   
title('Histogram for Jaccard Similarity Index')

Quantiles = [quantile(similarity, [0 .1 .25 .5 .75 .9 1]);[0 .1 .25 .5 .75 .9 1]]
mean_JIndex = mean(similarity)
median_JIndex = median(similarity)

%% Checking for pure clustering only!
% let's check how good are the cluster-segmentations!
figure
histogram(pure_similarity)
title('Histogram for Jaccard Similarity Index (Just "Pure" Clustering)')

Quantiles = [quantile(pure_similarity, [0 .1 .25 .5 .75 .9 1]);[0 .1 .25 .5 .75 .9 1]]
mean_JIndex = mean(pure_similarity)
median_JIndex = median(pure_similarity)
