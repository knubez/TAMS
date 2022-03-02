function sdist = signedDistancePointLine(point, line)
%DISTANCEPOINTLINE Minimum distance between a point and a line
%
%   D = distancePointLine(POINT, LINE)
%   Return the euclidean distance between line LINE and point POINT. 
%
%   LINE has the form: [x0 y0 dx dy], and POINT is [x y].
%
%   If LINE is N-by-4 array, result is N-by-1 array computes for each line.
%
%   If POINT is N-by-2, then result is computed for each point.
%
%   If both POINT and LINE are array, result is computed for each couple of
%   point and line, and is returned in a NP-by-NL array, where NP is the
%   number of points, and NL is the number of lines.
%
%
%   See also:
%   lines2d, points2d, distancePoints, distancePointEdge
%
   
% ------
% Author: David Legland
% e-mail: david.legland@inra.fr
% Created: 2017-04-14
% Copyright 2017 INRA - BIA-BIBS.

%   HISTORY:
%   2017-04-14 created based on distancePointLine

% direction vector of each line (row vectors)
vx = line(:, 3)';
vy = line(:, 4)';

% % direction vector of line "normal"
% nx = vy;
% ny = -vx;

% squared norm of direction vectors, with a check of validity
delta = (vx .* vx + vy .* vy);
invalidEdges = delta < eps;
delta(invalidEdges) = 1; 

% difference of coordinates between point and line origins
% (NP-by-NE arrays)
dx  = bsxfun(@minus, point(:, 1), line(:, 1)');
dy  = bsxfun(@minus, point(:, 2), line(:, 2)');

% compute position of points projected on the line, by using normalised dot
% product 
% (result is a NP-by-NL array) 
sdist = bsxfun(@rdivide, bsxfun(@times, dx, vy) - bsxfun(@times, dy, vx), delta);

% % ensure degenerated lines are correclty processed (consider the line
% % origin as closest point)
% pos(:, invalidEdges) = 0;
% 
% % compute distance between point and its projection on the line
% dist = hypot(bsxfun(@times, pos, vx) - dx, bsxfun(@times, pos, vy) - dy);




