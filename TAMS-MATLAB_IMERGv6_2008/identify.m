function [ data219, data235,data_day ] = identify(kiddy,i,data_day,filetoload,filebasename)
%CLASSIFY(filename)
%DESCRIP
%---------------------------------
%Identification of systems and classification between organized and convective
% 1. Converts ch9 counts (channel 10.8 micron, VIRS) from MSG to radiance to  brightness temperature
% 2. Option to plot original data
% 3. Creates contours of 235K, calculates areas and centroids
% 4. Creates contours of 219K, calculates areas (km)
% 5. Checking the points inside each 219 contour using mesh grid values and calculates average temp, standard deviation of each 219 area
% 6. If area is greater AND inside a 235 contour it is classified as  convective , else is DSL
% 7. Calculates centroid and areas (km using rectangle coord) for both convective and DSL.
%    Also calculates eccentricity and area ratios (A219/A235), and indices of contours outside each 219 K
% 8. Save mat files for each day --> thirdtrack.m ( whateverversion of track is in use at the moment) (SYSTEM TRACKING)
%

%HIST
%Created July/ 25/2017  by Kelly M. Nunez Ocasio  

% these could be function arguments in the future 
%Large domain
%xlims = [-76,60]; 
%ylims = [-5,50];

% Revised paper
xlims = [-45 60];
ylims = [-5  30];

% First paper
% xlims = [-35 45];
% ylims = [0  30];

% xlims = [-40,35];
% ylims = [-5,23];

% xlims = [-8,8];
% ylims = [-12,17];

load coastlines
coast = load('coast.mat');
borders('countries','nomap','k')
axis tight

%% initialize output files in case no data
data219 = struct([]);
data235 = struct([]);
%% Open netCDF file and load the things

ncid = netcdf.open(filetoload);

%Getting dates from filebasename
year=str2double(filebasename(59:62));
month=str2double(filebasename(63:64));
day=str2double(filebasename(65:66));
hour=str2double(filebasename(67:68));
minutes=str2double(filebasename(69:70));
seconds=str2double(filebasename(71:72));

stringdate=datetime(year,month,day,hour,minutes,seconds);

% Get values for each variable SEVIRI channel 10.8 which is channel 9 for years 2006 and 2007
%Lat = netcdf.getVar(ncid,140); %degress
%Lon = netcdf.getVar(ncid,141); %degrees
%ch9 = netcdf.getVar(ncid,137); % counts



% Get values fo reach variable SEVIRI channel 10.8 for rest of years
varidlat = netcdf.inqVarID(ncid,'lat');
varidlon = netcdf.inqVarID(ncid,'lon');
varidch9 = netcdf.inqVarID(ncid,'ch9');

Lat = netcdf.getVar(ncid,varidlat); % degrees
Lon = netcdf.getVar(ncid,varidlon); % degrees
ch9 = netcdf.getVar(ncid,varidch9); % counts/ radiance  
% convert to double precision
Lat = double(Lat);
Lon = double(Lon);
Lon(Lon== -999)=NaN;
Lat(Lat== -999)=NaN;

%% Saving Hour , MInutes and second inside data_day structure
data_day(i).year=filebasename(59:62);
data_day(i).month=filebasename(63:64);
data_day(i).day=filebasename(65:66);
data_day(i).hour =filebasename(67:68);
data_day(i).minutes =filebasename(69:70);
data_day(i).seconds = filebasename(71:72);


% Attributes associated with variable ch9
scale_factor = ncreadatt(filetoload,'ch9','scale_factor');
add_offset = ncreadatt(filetoload,'ch9','add_offset');

% Converting counts to radiance & accounting for scaling and offset
R= add_offset + (ch9 * scale_factor);

% Converting from Radiance to Brightness Temp
Temp=(( ( 1.43877 ) * ( 930.659 ) )./log ( ( ( (1.19104*10^(-5) ) * ( (930.659)^3 ) )./double(R) ) + 1 )-0.627)./0.9983;
TempB=real(Temp);
%% Plot original

%     f1=figure;
%     Lon(Lon== -999)=NaN;
%     Lat(Lat== -999)=NaN;
%     hold all;
%
%     %plotm(coastlat,coastlon,'Color','k')
%     hold all;
%     pcolor(Lon, Lat, TempB);
%     colormap jet
%     colorbar
%     caxis([180 300]);
%     shading flat
%     alpha(.5)
%     t=datetime(stringdate);
%     str=string(t);
%     text(-7,12.3,strcat({str},{' '},{'UTC'}));
%     hold on;
%     borders('countries','nomap','k')
%     % plot(lat,long,'black','Linewidth', 2)
%
%     xlabel('Longitude (degrees)');
%     ylabel('Latitude (degrees)');
%     xlim(xlims);
%     ylim(ylims);

%   figname=['/quoll/s0/hlh189/public/kmn18/graduateresearch/outputfiles/original_',num2str(year),num2str(month),num2str(day),num2str(hour),num2str(minutes) ];
%   saveas(f1,figname, 'png');

%% contour at the 219 and 235 levels and determine the things
% system identification
% c :  contains information : extract individual contours from the ContourMatrix c
% and compute areas and centroids of each closed contour

f2 = figure('visible','off');
cm = [0 0 0];
colormap(cm)
pcolor(Lon, Lat, TempB);
hold on 
% restrict to box region
xlim(xlims);
ylim(ylims);

set(gca, 'XTick', [])
set(gca, 'YTick', [])

imagefname = ['tmp_', data_day(i).year, data_day(i).month, data_day(i).day,...
data_day(i).hour, '.png'];
export_fig(f2, imagefname, '-r400');
Image = imread(imagefname);  % load the binary image data
delete(imagefname)  % Deleting the image each loop

% Location to obtain lat and lon from image
% note: ylims switched because of the way the image is read in
Lon1 = linspace(xlims(1), xlims(2), size(Image,2)); %for the image we use the # of columns to obtain longitudes
Lat1 = linspace(ylims(2), ylims(1), size(Image,1)); %for the image we use the # of rows to obtain latitudes % Linspace creates a vector

swath_true = ~Image;  % 2-D array describing which grid points are inside the swath
BW=edge(swath_true);


%% contour at the 219 and 235 levels and determine the things
% system identification
% c :  contains information : extract individual contours from the ContourMatrix c
% and compute areas and centroids of each closed contour
%    extraction code from: https://www.mathworks.com/matlabcentral/fileexchange/38863-extract-contour-data-from-contour-matrix-c/content/contourdata.m

% Mesh grid to plot TempB and locations together
[x1, y1] = meshgrid(Lon1, Lat1);
%figure;
%plot(x1(BW),y1(BW));
%size(x1(BW))
%size(x1(swath_true))
% Using location of interest to interpolate to allow for easier contouring
indexof_box = find(Lon>=xlims(1) & Lon<=xlims(2) & Lat>=ylims(1) & Lat<=ylims(2));
F=scatteredInterpolant(Lon(indexof_box),Lat(indexof_box),TempB(indexof_box));

% Finding points inside real data within location of interest 
lat=Lat(indexof_box);
lon=Lon(indexof_box);
temp=TempB(indexof_box);


% mask areas outside of the swath
x1(~ swath_true) = NaN;
y1(~ swath_true) = NaN;

% Plotting  data with contours
f3 = figure('visible','off');

[c, ~]=contour(x1,y1, F(x1,y1), [235 235], 'Linewidth', 4);
hold on;
[c1, ~]= contour(x1,y1, F(x1,y1), [219 219], 'Color','green', 'Linewidth',3.5); 
%t=datetime(stringdate);
%str=string(t);
%text(-7,12.3,strcat({str},{' '},{'UTC'}));
hold on;
borders('countries','nomap','k')
xlabel('Longitude (degrees)');
ylabel('Latitude (degrees)');
xlim(xlims);
ylim(ylims);

% ----------------------------------------------------------
% Areas of 235
tol = 1e-12;
k = 1;     % contour line number 235
col_index = 1;   % index of column containing contour level and number of points
while col_index < size(c,2); % while less than total columns in c
    
    data235(k).level = c(1,col_index);
    data235(k).numel = c(2,col_index);
    
    % extract x and y vertex points from the contourset
    idx = col_index+1:col_index+c(2,col_index);
    x = c(1,idx)';
    y = c(2,idx)';
    
    data235(k).isopen = abs(diff(c(1,idx([1 end])))) > tol || ...
        abs(diff(c(2,idx([1 end]))))>tol;  % is the contour closed?

    
    [x, y] = poly2cw(x, y);  % put in clockwise order
    
    % use polyarea() to calculate areas and save vertex points
    data235(k).area1 = polyarea(x, y);
    data235(k).x = x;
    data235(k).y = y;
    
    % changing x(lon) to km and calculate area
    xdist = 6371*x*(pi/180) .* cosd(y);  % using a equal rectangular projection
    ydist = 6371*y*(pi/180);
    data235(k).area_km = polyarea(xdist, ydist);
    
    % calculate centroids and alternative area calculation
    % the following from: https://www.mathworks.com/matlabcentral/fileexchange/7844-geom2d/content/geom2d/polygons2d/polygonCentroid.m
    N = length(x);
    iNext = [2:N 1];
    common = x .* y(iNext) - x(iNext) .* y;
    sx = sum((x + x(iNext)) .* common);  % compute cross products
    sy = sum((y + y(iNext)) .* common);
    area = sum(common) / 2;
    centroid = [sx sy] / 6 / area;
    data235(k).area2 = area;
    data235(k).centroid_x = centroid(1);
    data235(k).centroid_y = centroid(2);
    % Calculating centroids using rectangular projection (km) and thus xdist and dist
    N1 = length(xdist);
    iNext1 = [2:N1 1];
    common1 = xdist .* ydist(iNext1) - xdist(iNext1) .* ydist;
    sx1 = sum((xdist + xdist(iNext1)) .* common1);  % compute cross products
    sy1 = sum((ydist + ydist(iNext1)) .* common1);
    area1 = sum(common1) / 2;
    centroid1 = [sx1 sy1] / 6 / area1;
    data235(k).area3 = area1;
    data235(k).centroid_xdist = centroid1(1);
    data235(k).centroid_ydist = centroid1(2);
    
    
    % Calculating eccentricity using semi major and semi minor axis formula
    %  data235(k).dist_center=sqrt((centroid(1)-x).^2+(centroid(2)-y).^2);
    %  data235(k).semimajor=max(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2));
    %  data235(k).semiminor=min(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2));
    %  data235(k).eccentricity=sqrt(1-((min(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2))).^2/(max(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2))).^2));
    
    % plot centroid
    %plot(centroid(1), centroid(2), 'k.', ...
    %'MarkerSize', 20)
    
    %% Obtaining the points inside of each 235K edge
    %     [inside]=inpolygon(x1,y1,x,y);
    %     Latinside=y1(inside);
    %     Loninside=x1(inside);
    %     Tempinside=F(Loninside,Latinside);
    % data235(k).Latinside=Latinside;
    % data235(k).Loninside=Loninside;
    % data235(k).Tempinside=Tempinside;
    
    
    
    k = k + 1;
    col_index = col_index + c(2,col_index) + 1;
end
 
% ----------------------------------------------------------
% Area of 219
tol2 = 1e-12;
kk = 1;     % contour line number 219
col_index2 = 1;   % index of column containing contour level and number of points
while col_index2 < size(c1,2); % while less than total columns in c
    
    data219(kk).level = c1(1,col_index2);
    data219(kk).numel = c1(2,col_index2);
    data219(kk).kids=kiddy;
    % extract x and y vertex points from the contourset
    idx=col_index2+1:col_index2+c1(2,col_index2);
    x = c1(1,idx)';
    y = c1(2,idx)';
    
    % Find lat = to y and lon = to x
    data219(kk).isopen = abs(diff(c1(1,idx([1 end])))) > tol2 || ...
        abs(diff(c1(2,idx([1 end]))))>tol2;  % is the contour open?
    
    
    [x, y] = poly2cw(x, y);  % put in clockwise order
    
    data219(kk).area1 = polyarea(x, y);
    data219(kk).x = x;
    data219(kk).y = y;
    
    %% Obtaining the points inside of each 219K edge
    [inside]=inpolygon(x1,y1,x,y);
    Latinside=y1(inside);
    Loninside=x1(inside);
    Tempinside=F(Loninside,Latinside);
    % data219(kk).Latinside=Latinside;
    %  data219(kk).Loninside=Loninside;
    %  data219(kk).Tempinside=Tempinside;
    
    
    %% Average temperature calculation based on number of points (pixels) (Tsakraklides and Evans 2003)
    meantempreal = sum(Tempinside) / length(Tempinside);
    data219(kk).meantempreal= meantempreal;
    %% Standard deviation of average temperatures (<219
    
    %data219(kk).stdreal = sqrt(sum((Tempinsidereal-meantempreal).^2) / length(Tempinsidereal));
    data219(kk).std1real = std(Tempinside);
    
    
    
    
    % changing x(lon) to km and calculate area
    xdist = 6371*x*(pi/180) .* cosd(y); % using a equal rectangular projection
    ydist = 6371*y*(pi/180);
    data219(kk).area_km = polyarea(xdist, ydist);
    
    
    
    % IDENTIFYING IF AREA IS DSL OR CONVECTIVE (A>=4000 IT IS CLASSIFIED AS CONVECTIVE)
    % Cheking if 219 are inside 235 is to be able to  sort deep,convective clouds from warmer non precipitating clouds
    [isin235, inds235outside] = in235(x, y, data235);
    data219(kk).inds235outside = inds235outside;  % save indices of the 235 contours that the 219 is inside
    
    if polyarea(xdist, ydist) >= 4000 && isin235
        data219(kk).Class = 'Convective';
    else
        data219(kk).Class = 'DSL';
    end
    
    % -----------------------------------------
    
    
    
    % calculate centroids and alternative area calculation
    % the following from: https://www.mathworks.com/matlabcentral/fileexchange/7844-geom2d/content/geom2d/polygons2d/polygonCentroid.m
    N = length(x);
    iNext = [2:N 1];
    common = x .* y(iNext) - x(iNext) .* y;
    sx = sum((x + x(iNext)) .* common);  % compute cross products
    sy = sum((y + y(iNext)) .* common);
    area = sum(common) / 2;  % area and centroid
    centroid = [sx sy] / 6 / area;
    data219(kk).area2 = area;
    data219(kk).centroid_x = centroid(1);
    data219(kk).centroid_y = centroid(2);
    
    % Calculating centroids using rectangular projection (km) and thus
    % xdist and dist
    N1 = length(xdist);
    iNext1 = [2:N1 1];
    common1 = xdist .* ydist(iNext1) - xdist(iNext1) .* ydist;
    sx1 = sum((xdist + xdist(iNext1)) .* common1);  % compute cross products
    sy1 = sum((ydist + ydist(iNext1)) .* common1);
    area1 = sum(common1) / 2;
    centroid1 = [sx1 sy1] / 6 / area1;
    data219(kk).area3 = area1;
    data219(kk).centroid_xdist = centroid1(1);
    data219(kk).centroid_ydist = centroid1(2);
    
    % Calculating eccentricity using semi major and semi minor axis formula
    %  data219(kk).dist_center=sqrt((centroid(1)-x).^2+(centroid(2)-y).^2);
    %  data219(kk).semimajor=max(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2));
    %  data219(kk).semiminor=min(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2));
    %  data219(kk).eccentricity=sqrt(1-((min(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2))).^2/(max(sqrt((centroid(1)-x).^2+(centroid(2)-y).^2))).^2));
    
    
    % Setting a name / identifier for each contour
    % data219(kk).name=randseq(5);
    %plot(centroid(1), centroid(2), 'r.', ...
    % 'MarkerSize', 20)
    
    
    
    kk = kk + 1;
    col_index2 = col_index2 + c1(2,col_index2) + 1;
    kiddy=1+kiddy;
    
end


% area ratios
for p = 1:length(data235)
    area235 = data235(p).area_km;
    
    areas219 = [];
    for j = 1:length(data219)
        if data219(j).inds235outside == p
            areas219(end+1) = data219(j).area_km;
        end
    end
    
    data235(p).ratios_individual = areas219 ./ area235; % individual ratios for 219 areas inside a specific 235 area
    
    data235(p).ratios_sum = sum(areas219) / area235; % Sum(A219)/A235 ratios for sum of all A219 inside  respective A235
    
    data235(p).sum_area219 = sum(areas219); % sum of the 219 areas inside each 235
    
end

end



function [isin235, inds235outside] = in235(x219, y219, data235)
%IN235 check if a singular 219 contour is inside of any of the 235 contours

isin235 = 0;  % false
inds235outside = [];
for h = 1:length(data235)
    
    x235 = data235(h).x; y235 = data235(h).y;
    
    if sum(sum(inpolygon(x219, y219, x235, y235)))  % note: [in, on] = inpolygon(...
        isin235 = 1;  % true
        inds235outside(end+1) = h;
    end
end
end


