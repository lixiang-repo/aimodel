select
    *, 
    cast(click_type as float) as label 
from dsp.push_all_v1
where year='{y}' and month='{m}' and day='{d}' and click_type='1'