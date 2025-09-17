
; Sample nodes and edges with mutual friendships
(person John)
(person Mary)
(person Bob)
(person Alice)
(person Charlie)

(age John 30)
(age Mary 25)
(age Bob 35)
(age Alice 28)
(age Charlie 42)

; Mutual friendships
(friend John Mary)
(friend Mary John)  ; This makes it mutual

(friend Mary Bob)
(friend Bob Mary)   ; This makes it mutual

(friend Alice Charlie)
; Charlie → Alice is missing, so this is one-way

; One-way friendships
(friend Bob John)   ; Bob → John (but John → Mary, not mutual)
(friend Charlie John) ; Charlie → John (one-way)

(parent John Bob)
(parent Mary Bob)
(parent Alice Charlie)

(location John "New York")
(location Mary "Boston")
(location Bob "Chicago")
