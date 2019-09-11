%
% SCRIPT: FINDTRIANGLES
%
%   Download a graph from Sparse Matrix Collection and count the number of
%   triangles.
%

%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - January 21, 2019
%


%% CLEAN-UP

clear
close all


%% PARAMETERS

basePath  = 'http://sparse.tamu.edu/mat';
groupName = 'DIMACS10';
matName   = 'auto'; % auto|great-britain_osm|delaunay_n22

%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD INPUT GRAPH

fprintf( '...loading graph...\n' ); 
fprintf( '   - %s/%s\n', groupName, matName )

fileName = [groupName '_' matName '.mat'];

if ~exist( fileName, 'file' )
  fprintf('   - downloading graph...\n')
  fileName = websave( fileName, ...
                      [basePath filesep groupName filesep matName '.mat'] );
  fprintf('     DONE\n')
end

ioData  = matfile( fileName );
Problem = ioData.Problem;

% keep only adjacency matrix (logical values)
A = Problem.A > 0; 
clear Problem;

fprintf( '   - DONE\n');

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 
ticCnt = tic;

nT = full( sum( sum( A^2 .* A ) ) / 6 );

fprintf( '   - DONE: %d triangles found in %.2f sec\n', nT, toc(ticCnt) );


%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);


