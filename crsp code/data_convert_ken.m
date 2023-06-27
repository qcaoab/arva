
%
%      in real indexes file:
%  Column
%  1     Time (the date as a decimal number, e.g. 1926.75 for Sept 1926)
%  2     CPI (the CPI index from CRSP)
%  3     T30 (30-day T-bill real total return index)
%  4     T90 (90-day T-bill real total return index)
%  5     B10 (back-filled real 10-year bond total return index),
%  6     VWD (real value-weighted with distributions total return stock index)
%  7     VWX (real value-weighted without distributions total return stock index),
%  8     EWD (real equal-weighted with distributions total return stock index),
%  9     EWX (real equal-weighted without distributions total return stock index)

%       All indexes start at 100 for Time=1926 (which is the start of 1926/end of 1925)
   start_time = 1926;   % Jan 1
   end_time = 2022;     % Jan 1
   filename = 'real_indexes_2021.csv';

%


data_in = importdata(filename);

number = length( data_in.data(:,1) );



% assume equally spaced monthly data

index = data_in.data(:,6);   % cap wted CRSP
time = data_in.data(:,1);

fid =  fopen('Real_CapWT_CRSP_Feb_2022.dat', 'w' );

fprintf(fid, 'Real CapWT CRSP: 1926:1-2021:12 (deflated) as of February, 2022 \n', 'w');

fprintf(fid, 'Time  IndexValue NumRecords %d\n', number);

for j=1:number
  fprintf(fid, '%15.8f    %15.12g \n', time(j),  index(j));
end

fclose(fid);


index = data_in.data(:,3);   % 30 day T-bill

fid =  fopen('Real_30_day_T_bill_Feb_2022.dat', 'w' );
fprintf(fid, 'Real 30 day T-bill: 1926:1-2021:12 (deflated) as of Feb, 2022 \n', 'w');

fprintf(fid, 'Time  IndexValue NumRecords %d\n', number);
   for j=1:number
  fprintf(fid, '%15.8f    %15.12g \n', time(j),  index(j));
end

fclose(fid);


index = data_in.data(:,5);   % 10 yr bond
fid =  fopen('Real_10yr_bond_Feb_2022.dat', 'w' );
fprintf(fid, 'Real  10yr Treasury: 1926:1-2021:12 (deflated) as of Feb, 2022 \n', 'w');

fprintf(fid, 'Time  IndexValue NumRecords %d\n', number);
   for j=1:number
  fprintf(fid, '%15.8f    %15.12g \n', time(j),  index(j));
end

fclose(fid);


index = data_in.data(:,8);   % Equal wt Crsp
fid =  fopen('Real_EqualWt_CRSP_Feb_2022.dat', 'w' );
fprintf(fid, 'Real  EqualWT CRSP: 1926:1-2021:12 (deflated) as of Feb, 2022 \n', 'w');

fprintf(fid, 'Time  IndexValue NumRecords %d\n', number);
   for j=1:number
  fprintf(fid, '%15.8f    %15.12g \n', time(j),  index(j));
end

fclose(fid);


index = data_in.data(:,2);   % CPI
fid =  fopen('CPI_Feb_2022.dat', 'w' );
fprintf(fid, 'CPI: 1926:1-2021:12 (deflated) as of Feb, 2022 \n', 'w');

fprintf(fid, 'Time  IndexValue NumRecords %d\n', number);
   for j=1:number
  fprintf(fid, '%15.8f    %15.12g \n', time(j),  index(j));
end

fclose(fid);


