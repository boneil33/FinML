// function to compute ratio roll adj
{[futroot;futsuffix;num_contracts;days_to_notice;start_date]
    
    
    

    /basic
//     futroot:"ED"
//     futsuffix:" Comdty"
//     num_contracts:8
//     days_to_notice:-8
//     start_date: 2000.01.01
    /removeYK: {`$(" "vs'string x)[;0]};
    bdays: >>bday array
    syms: `$futroot,/:(string 1+til num_contracts+1);
    symsWithYK: `$(string syms),\:futsuffix;
    raw: ??raw data from bbg
    data: update gen_sym: {`$(" "vs'string x)[;0]} sym, px_last:"F"$px_last, sym: fills `$fut_cur_gen_ticker from raw;
    data: `date`gen_sym xasc data;
    data: delete from data where (num_contracts+1)>(count;sym) fby date;

    / distinct historical tickers
    distTickers: select ticker: distinct sym from data;
    distTickers: update tickerWithYK: (`$(string ticker),\:futsuffix) from distTickers;
    / get roll dates and 10bd prior
    rollDates: update sym: {`$(" "vs'string x)[;0]} sym, roll_dt: ??busday shift func from 
        update shift:days_to_notice, fut_notice_first: "D"$fut_notice_first from ??bbg first notices

    full_data:`roll_dt`sym xasc select gen_sym, date, sym, px_last, roll_dt from ej[`sym;select sym, roll_dt from rollDates;data];
    / create your own gen_sym
    full_rank_data: update custom_gen_sym: `$(futroot,/:string sym_rank+1) from (`date`sym_rank xasc update sym_rank: rank roll_dt by date 
        from (select from full_data where roll_dt>=date));
    distinctRollDates: select distinct roll_dt from full_rank_data;
    front_month_data: select date, join_rank:sym_rank+1, custom_gen_sym1: custom_gen_sym, px_last1: px_last, sym1:sym, roll_dt 
        from full_rank_data where date in (exec roll_dt from distinctRollDates);
    back_month_data: select date, join_rank:sym_rank, custom_gen_sym2: custom_gen_sym, px_last2: px_last, sym2:sym 
        from full_rank_data where date in (exec roll_dt from distinctRollDates);

    / need to join the previous roll date (i.e. the date the front contract became the front)
    distinctRollDates: update idx: rank roll_dt, next_idx: 1+ rank roll_dt from distinctRollDates;
    rollLookup: select roll_dt, prev_roll_dt from distinctRollDates lj `idx xkey `prev_roll_dt`next_idx`idx xcol distinctRollDates;


    / roll_diff for each date where ED1 roll_dt = date
    rollDiffs: `date xdesc `join_rank xasc update roll_diff: px_last2-px_last1 from front_month_data 
        lj `date`join_rank xkey back_month_data; /ej[`date`join_rank; back_month_data; front_month_data]
    rollRatios: `date xdesc `join_rank xasc update roll_ratio: px_last2%px_last1 from front_month_data 
        lj `date`join_rank xkey back_month_data;
    / need cumulative backfill roll diffs, makes sense because curve is upward sloping on average for rates as dummy check
    / compute cumulative sum by join rank into an asof join- just double check the asof join logic
    rollDiffs: update cum_roll: sums roll_diff by join_rank from rollDiffs;
    rollRatios: update cum_roll: prds roll_ratio by join_rank from rollRatios;
    / join the previous roll date (i.e. when that contract became the front). remember sym1 is still the front on roll dt
    rollDiffs: rollDiffs lj `roll_dt xkey rollLookup;
    rollRatios: rollRatios lj `roll_dt xkey rollLookup;

    / left join and bfills, asof join doesnt do exactly what we need
    backRolled:full_rank_data lj `custom_gen_sym`date xkey select date, custom_gen_sym: custom_gen_sym1, cum_roll from rollDiffs;
    backRolled: update fills cum_roll by custom_gen_sym from `date xdesc `sym_rank xasc select from backRolled where date >= start_date;
    backRolled: update px_last_adj: px_last+0^cum_roll from backRolled;
    / same for ratio
    backRolledRatio:full_rank_data lj `custom_gen_sym`date xkey select date, custom_gen_sym: custom_gen_sym1, cum_roll from rollRatios;
    backRolledRatio: update fills cum_roll by custom_gen_sym from `date xdesc `sym_rank xasc 
        select from backRolledRatio where date>=start_date;
    backRolledRatio: update px_last_adj: px_last*1^cum_roll, cum_roll:1^cum_roll from backRolledRatio;
    
    select date, custom_gen_sym, px_last_adj, cum_roll from backRolledRatio where sym_rank<num_contracts
    }
