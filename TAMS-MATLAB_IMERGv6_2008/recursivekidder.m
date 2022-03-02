function [list]=recursivekidder(i,list,newsystems)
%organizer-name of function
%----------------
% 1. This function takes the newsystem structure form overlap_with_precip_new and using eacj row kids,
%    will return a list of all the kids connected trhgouh recurvsrvuly looping through each kids .


% HIST
% Created January  16th 2017 by Kelly M. Nunez Ocasio 

% --------------------------------------------------------------------------------------------------------------------

list=[list,i];
%disp([' preparent ', num2str(i)])

for parent=1:length(newsystems(i).globalparent)
    
    
    if ~ismember(newsystems(i).globalparent(parent),list)
        list=recursivekidder(newsystems(i).globalparent(parent),list,newsystems);
    end
end
%disp([' tween ', num2str(i)])
for kid=1:length(newsystems(i).globalkids)
    if ~ismember(newsystems(i).globalkids(kid),list)
        list=recursivekidder(newsystems(i).globalkids(kid),list,newsystems);
    end
    
end
%disp([' postkid ', num2str(i)])


%% Original
% list=[list,i];
% for kids=2:length(newsystems(i).kids)
%     list=recursivekidder(newsystems(i).kids(kids),list,newsystems);
% end
