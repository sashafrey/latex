function [resultString] = RepeatSymbol(symbol, times)
%RepeatSymbol: creates string, that consists of only 1 symbol, that is
%repeated times.
%
%   Input:
%   symbol: symbol to repeat
%   times: number of times to repeat symbol
%
%   Output:
%   resultString: string of repeated times times symbol
%
%   Example:
%   RepeatSymbol('a', 5) => 'aaaaa'
resultString = '';
for n = 1: times
    resultString = sprintf('%s%c', resultString, symbol);
end
return;