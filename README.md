# pydemo-flask

## This is a demo version of Hair color analyzer of a picture

Using Python 3 and OpenCV

First of all, this is a demo version, created in 3 days, so please do not expect it working perfectly :)

There are following limitations:

It only works with frontal face pictures (passport-styled photos are the best source). It may be extended in the future.

It only uses a single color for matching. Obviously, some patterns have two or more shades of colors, so the match is not perfect in such situations. This limitation is quite easy to remove in the future, using two or more colors.

It is VE-E-E-E-RY slow on pythonanywhere.com! (at least at it's free account) On my local machine it is much much quicker. Seconds... I even had to change the face detection method to from slower-but-more-presize to a faster-but-less-presize when deploying to pythonanywhere.

Thank you for looking! 

Working version on pythonanywhere:
http://vallka.pythonanywhere.com/
