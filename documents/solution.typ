= Solution Aspects
Fundamentally there are also short-term temporal dependencies, i.e. given that many people have used a shared bicycle to reach their workplace at a specific day, they are more likely to also return to their home.
Another dependency arising is the connectivity of stations, i.e. if the demand to take a bike rises a one station, the demand to dock a bike at a *connected* station (i.e. there occur frequent rides between these two stations) should rise.

== Implementation
=== Masking
For training, we need to mask out every section where the demand is at its daily max/min if 
- this max/min is reached at elongated period of the day 
- is not too far away from the capacity estimate (e.g. a station at the university may not reach its full capacity on weekends)