from Engine.PkmBaseStructures import PkmMove, PkmType, WeatherCondition, PkmStat, PkmEntryHazard, PkmStatus

# Special Moves
SunnyDay = PkmMove(0., PkmType.FIRE, "Sunny Day", lambda v: v.set_weather(WeatherCondition.SUNNY))
RainDance = PkmMove(0., PkmType.WATER, "Rain Dance", lambda v: v.set_weather(WeatherCondition.RAIN))
Hail = PkmMove(0., PkmType.ICE, "Hail", lambda v: v.set_weather(WeatherCondition.HAIL))
Sandstorm = PkmMove(0., PkmType.ROCK, "Sandstorm", lambda v: v.set_weather(WeatherCondition.SANDSTORM))
NastyPlot = PkmMove(0., PkmType.DARK, "Nasty Plot", lambda v: v.set_status(PkmStat.ATTACK, 2, 0))
BulkUp = PkmMove(0., PkmType.FIGHT, "Bulk Up", lambda v: v.set_status(PkmStat.ATTACK, 2, 0))
CalmMind = PkmMove(0., PkmType.PSYCHIC, "Calm Mind", lambda v: v.set_status(PkmStat.DEFENSE, 2, 0))
IronDefense = PkmMove(0., PkmType.STEEL, "Iron Defense", lambda v: v.set_status(PkmStat.DEFENSE, 2, 0))
StringShot = PkmMove(0., PkmType.BUG, "String Shot", lambda v: v.set_status(PkmStat.DEFENSE, -1, 1))
Spikes = PkmMove(0., PkmType.GROUND, "Spikes", lambda v: v.set_entry_hazard(PkmEntryHazard.SPIKES, 1))
SweetKiss = PkmMove(0., PkmType.FAIRY, "Sweet Kiss", lambda v: v.set_status(PkmStatus.CONFUSED, 1))
Poison = PkmMove(0., PkmType.POISON, "Poison", lambda v: v.set_status(PkmStatus.POISONED, 1))
Spore = PkmMove(0., PkmType.GRASS, "Spore", lambda v: v.set_status(PkmStatus.SLEEP, 1))
ThunderWave = PkmMove(0., PkmType.ELECTRIC, "Thunder Wave", lambda v: v.set_status(PkmStatus.PARALYZED, 1))
Recover = PkmMove(0., PkmType.NORMAL, "Recover", lambda v: v.set_recover(80.))
Roost = PkmMove(0., PkmType.FLYING, "Roost", lambda v: v.set_recover(80.))
DragonRage = PkmMove(0., PkmType.DRAGON, "Dragon Rage", lambda v: v.set_fixed_damage(100.))
NightShade = PkmMove(0., PkmType.GHOST, "Night Shade", lambda v: v.set_fixed_damage(100.))