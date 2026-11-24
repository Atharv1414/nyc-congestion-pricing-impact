-- Pickups and drop-offs by zone, day and fare type, including flex-fare trips, for the demand analysis.
select
    cast(picked_up_at as date) as day,
    pu_zone,
    do_zone,
    case when payment_type = 0 then 'flex' else 'metered' end as fare_type,
    count(*) as trips,
    sum(fare) as fare,
    count(*) filter (where cbd_fee > 0) as cbd_fee_trips
from flagged
-- a trip with a bad timestamp is still a trip, so only checks 1 to 4 apply here
where left(status, 1) not in ('1', '2', '3', '4')
group by all;
