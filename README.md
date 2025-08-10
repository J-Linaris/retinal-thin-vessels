# retinal_thin_vessels
FUNCTIONS:

    get_thin_vessels: returns a filtered image containing only the vessels whose 
                      width is less than or equal to the passed "ceil" value. 
                      Also, it supports mask_type argument, so you can pass the 
                      probabilities score mask produced by Sigmoid directly
                      to the function and it will show the filtered mask.
