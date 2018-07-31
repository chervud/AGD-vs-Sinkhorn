
function [newImage]=my_im_resize(row,col,oldImage)

%# Initializations:

scale = [row col];              %# The resolution scale factors: [rows columns]
oldSize = size(oldImage);                   %# Get the size of your image
newSize = max(floor(scale.*oldSize(1:2)),1);  %# Compute the new image size

%# Compute an upsampled set of indices:

rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5),oldSize(1));
colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5),oldSize(2));

%# Index old image to get new image:

newImage = oldImage(rowIndex,colIndex,:);

end