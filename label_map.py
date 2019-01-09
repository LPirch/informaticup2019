
gtsrb_map = {
	16:	'Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse von 3,5t',
	7:	'Zulässige Höchstgeschwindigkeit (100)',
	5:	'Zulässige Höchstgeschwindigkeit (80)',
	10:	'Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t',
	9:	'Überholverbot für Kraftfahrzeuge aller Art',
	1:	'Zulässige Höchstgeschwindigkeit (30)',
	2:	'Zulässige Höchstgeschwindigkeit (50)',
	4:	'Zulässige Höchstgeschwindigkeit (70)',
	0:	'Zulässige Höchstgeschwindigkeit (20)',
	38:	'Rechts vorbei',
	18:	'Gefahrenstelle',
	17:	'Verbot der Einfahrt',
	33:	'Ausschließlich rechts',
	21:	'Doppelkurve (zunächst links)',
	25:	'Baustelle',
	11:	'Einmalige Vorfahrt',
	31:	'Wildwechsel',
	40:	'Kreisverkehr',
	27:	'Fußgänger',
	20:	'Kurve (rechts)',
	12:	'Vorfahrt',
	32:	'Ende aller Streckenverbote',
	15:	'Verbot für Fahrzeuge aller Art',
	35:	'Ausschließlich geradeaus',
	6:	'Ende der Geschwindigkeitsbegrenzung (80)',
	8:	'Zulässige Höchstgeschwindigkeit (120)',
	3:	'Zulässige Höchstgeschwindigkeit (60)',
	19:	'Kurve (links)',
	23:	'Schleudergefahr bei Nässe oder Schmutz',
	13:	'Vorfahrt gewähren',
	42:	'Ende des Überholverbotes für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t',
	14:	'Stoppschild',
	41:	'Ende des Überholverbotes für Kraftfahrzeuge aller Art',
	29:	'Fahrradfahrer',
	22:	'Unebene Fahrbahn',
	39:	'Links vorbei'
}

# make dict bidirectional
gtsrb_map.update(dict([reversed(item) for item in gtsrb_map.items()]))

remote_map = {
	0:	'Verbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse von 3,5t',
	1:	'Zulässige Höchstgeschwindigkeit (100)',
	2:	'Zulässige Höchstgeschwindigkeit (80)',
	3:	'Überholverbot für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t',
	4:	'Überholverbot für Kraftfahrzeuge aller Art',
	5:	'Zulässige Höchstgeschwindigkeit (30)',
	6:	'Zulässige Höchstgeschwindigkeit (50)',
	7:	'Zulässige Höchstgeschwindigkeit (70)',
	8:	'Zulässige Höchstgeschwindigkeit (20)',
	9:	'Rechts vorbei',
	10:	'Gefahrenstelle',
	11:	'Verbot der Einfahrt',
	12:	'Ausschließlich rechts',
	13:	'Doppelkurve (zunächst links)',
	14:	'Baustelle',
	15:	'Einmalige Vorfahrt',
	16:	'Wildwechsel',
	17:	'Kreisverkehr',
	18:	'Fußgänger',
	19:	'Kurve (rechts)',
	20:	'Vorfahrt',
	21:	'Ende aller Streckenverbote',
	22:	'Verbot für Fahrzeuge aller Art',
	23:	'Ausschließlich geradeaus',
	24:	'Ende der Geschwindigkeitsbegrenzung (80)',
	25:	'Zulässige Höchstgeschwindigkeit (120)',
	26:	'Zulässige Höchstgeschwindigkeit (60)',
	27:	'Kurve (links)',
	28:	'Schleudergefahr bei Nässe oder Schmutz',
	29:	'Vorfahrt gewähren',
	30:	'Ende des Überholverbotes für Kraftfahrzeuge mit einer zulässigen Gesamtmasse über 3,5t',
	31:	'Stoppschild',
	32:	'Ende des Überholverbotes für Kraftfahrzeuge aller Art',
	33:	'Fahrradfahrer',
	34:	'Unebene Fahrbahn',
	35:	'Links vorbei'
}

# make dict bidirectional
remote_map.update(dict([reversed(item) for item in remote_map.items()]))

unknown_labels = {
	24: 'einseitig (rechts) verengte Fahrbahn',
	26: 'Lichtzeichenanlage',
	28: 'Kinder',
	30: 'Schnee- oder Eisglätte',
	34: 'Ausschließlich links',
	36: 'vorgeschriebene Fahrtrichtung (geradeaus und rechts)',
	37: 'vorgeschriebene Fahrtrichtung (geradeaus und links)'
}

# make dict bidirectional
unknown_labels.update(dict([reversed(item) for item in unknown_labels.items()]))
