-- Speed panel: origin-destination pair x week x time band. Mean log speed is kept rather than
-- total miles / total hours so that a few very long trips cannot move a cell.
select
    pu_zone,
    do_zone,
    cast(date_trunc('week', picked_up_at) as date) as week,
    case
        when dayofweek(picked_up_at) in (0, 6)
            then case when hour(picked_up_at) between 9 and 20 then 'weekend_peak' else 'overnight' end
        when hour(picked_up_at) between 5 and 20 then 'weekday_peak'
        else 'overnight'
    end as band,
    status = 'kept: metered' as metered,
    count(*) as trips,
    sum(ln(mph)) as sum_log_mph,
    sum(ln(miles)) as sum_log_miles,
    sum(miles) as miles,
    sum(hours) as hours
from flagged
where status like 'kept%'
group by all;
