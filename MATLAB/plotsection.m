function plotsection(size,pixellist,start,ter)


[row,col] = ind2sub(size,pixellist(start:ter))

c = minBoundingBox([col , row]')

s= [col , row]'

hold on, plot(s(1,:),s(2,:),'.'); 
hold on,   plot(c(1,[1:end 1]),c(2,[1:end 1]),'r','LineWidth',2); hold on
