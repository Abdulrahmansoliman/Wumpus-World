:- dynamic grid_size/1.
:- dynamic breeze/2.
:- dynamic no_breeze/2.
:- dynamic stench/2.
:- dynamic no_stench/2.
:- dynamic wumpus_dead/0.

coordinates(Coords) :-
    grid_size(N),
    findall((X, Y), (between(1, N, X), between(1, N, Y)), Coords).

adjacent(X, Y, NX, NY) :-
    grid_size(N),
    (NX is X + 1, NY is Y ; NX is X - 1, NY is Y ; NX is X, NY is Y + 1 ; NX is X, NY is Y - 1),
    NX >= 1, NY >= 1, NX =< N, NY =< N.

choose_wumpus(Coords, (WX, WY)) :-
    member((WX, WY), Coords),
    (WX, WY) \= (1, 1).

choose_pits([], []).
choose_pits([(1, 1) | Rest], Pits) :-
    choose_pits(Rest, Pits).
choose_pits([Coord | Rest], [Coord | PitsRest]) :-
    choose_pits(Rest, PitsRest).
choose_pits([_Coord | Rest], Pits) :-
    choose_pits(Rest, Pits).

has_adjacent_pit(X, Y, Pits) :-
    adjacent(X, Y, PX, PY),
    member((PX, PY), Pits).

has_adjacent_wumpus(X, Y, WX, WY) :-
    adjacent(X, Y, WX, WY).

respect_breeze(Pits) :-
    forall(breeze(X, Y), has_adjacent_pit(X, Y, Pits)),
    forall(no_breeze(X, Y), \+ has_adjacent_pit(X, Y, Pits)).

respect_stench(_) :-
    wumpus_dead,
    !.
respect_stench((WX, WY)) :-
    forall(stench(X, Y), has_adjacent_wumpus(X, Y, WX, WY)),
    forall(no_stench(X, Y), \+ has_adjacent_wumpus(X, Y, WX, WY)).

safe_start(Pits, (WX, WY)) :-
    \+ member((1, 1), Pits),
    (WX, WY) \= (1, 1).

consistent_model((WX, WY), Pits) :-
    safe_start(Pits, (WX, WY)),
    \+ member((WX, WY), Pits),
    respect_breeze(Pits),
    respect_stench((WX, WY)).

model((WX, WY), Pits) :-
    coordinates(Coords),
    choose_wumpus(Coords, (WX, WY)),
    choose_pits(Coords, Pits),
    consistent_model((WX, WY), Pits).

safe_in_model((WX, WY), Pits, (X, Y)) :-
    \+ member((X, Y), Pits),
    (wumpus_dead ; (WX, WY) \= (X, Y)).

safe_in_all_models(_, [], []).
safe_in_all_models(Coords, Models, Safe) :-
    include(is_safe_in_models(Models), Coords, Safe).

is_safe_in_models(Models, Coord) :-
    forall(member((WX, WY, Pits), Models), safe_in_model((WX, WY), Pits, Coord)).

provably_safe(Safe) :-
    coordinates(Coords),
    findall((WX, WY, Pits), model((WX, WY), Pits), Models),
    Models \= [],
    safe_in_all_models(Coords, Models, Safe),
    !.
provably_safe([]).
