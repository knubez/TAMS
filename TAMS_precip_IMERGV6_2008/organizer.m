function [Systems] = organizer(newsystems) 
%organizer-name of function
%----------------
% 1. This function takes the newsystem structure form overlap_with_precip_new and using
%    a recursive kidder which is a function that calls itself , it sorts all the systems by providing a list of its positions .


% HIST
% Created January  16th 2017 by Kelly M. Nunez Ocasio with collaboration of Dr Young

% --------------------------------------------------------------------------------------------------------------------
tic;
masterlist=1:length(newsystems);
k=0;
while length(masterlist)>0  % Loop over systems
    k=k+1;
    list=[];
    i = masterlist(1);
    list = recursivekidder(i,list,newsystems) ; % recursive function calls to loop over storms in a system
    list = unique(list);
    % story list away as kth element in an structure of arrays ****
    Systems(k).list=list;
    masterlist = setdiff(masterlist,list);

end
 
%disp('List for classification ended')
disp('Time organizer ends')
toc;





%% Original version 
% tic;
% masterlist=1:length(newsystems);
% k=0;
% while length(masterlist)>0  % Loop over systems
%     k=k+1;
%     list=[];
%     i = masterlist(1);
%     list = recursivekidder(i,list,newsystems) ; % recursive function calls to loop over storms in a system
%     list = unique(list);
%     % story list away as kth element in an structure of arrays ****
%     Systems(k).list=list;
%     masterlist = setdiff(masterlist,list);
% 
% end
% 
% disp('List for classification ended')
% disp('Tim organizer ends')
% toc;
























