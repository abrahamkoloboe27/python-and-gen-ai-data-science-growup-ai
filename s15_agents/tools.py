"""
S15 — Tools Definition
Définition des outils disponibles pour l'agent
"""

import re
from typing import Dict, Any, Callable
from datetime import datetime
import random


class Tool:
    """Classe représentant un outil utilisable par l'agent"""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Dict[str, str]
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
    
    def execute(self, **kwargs) -> str:
        """Exécuter l'outil avec les paramètres fournis"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Erreur lors de l'exécution de {self.name}: {str(e)}"
    
    def get_signature(self) -> str:
        """Obtenir la signature de l'outil pour le prompt"""
        params_str = ", ".join([f"{k}: {v}" for k, v in self.parameters.items()])
        return f"{self.name}({params_str}): {self.description}"


# ============================================================
# TOOL 1: WEB SEARCH (Stub)
# ============================================================

def web_search(query: str) -> str:
    """
    Simule une recherche web (stub pour démo)
    En production, utiliser une vraie API (SerpAPI, Brave Search, etc.)
    
    Args:
        query: Requête de recherche
        
    Returns:
        Résultats simulés
    """
    # Base de connaissances simulée
    knowledge = {
        "paris londres prix": "Le prix moyen d'un billet Paris-Londres est de 80-150€ selon la période.",
        "paris population": "Paris compte environ 2.2 millions d'habitants intra-muros.",
        "paris capitale": "Paris est la capitale de la France.",
        "paris hotel prix": "Les hôtels à Paris coûtent en moyenne 100-200€ par nuit.",
        "restaurant paris prix": "Un repas dans un restaurant parisien coûte en moyenne 15-40€.",
        "vol paris new york": "Un vol Paris-New York coûte entre 400€ et 1200€ selon la saison.",
        "eiffel tower height": "La Tour Eiffel mesure 330 mètres de hauteur.",
        "python langage": "Python est un langage de programmation créé par Guido van Rossum en 1991.",
    }
    
    query_lower = query.lower()
    
    # Chercher la meilleure correspondance
    for key, value in knowledge.items():
        if any(word in query_lower for word in key.split()):
            return f"Résultat de recherche: {value}"
    
    # Réponse par défaut
    return f"Résultat de recherche pour '{query}': Information non disponible dans la base de démonstration."


# ============================================================
# TOOL 2: CALCULATOR
# ============================================================

def calculator(expression: str) -> float:
    """
    Évalue une expression mathématique de manière sécurisée
    
    Args:
        expression: Expression mathématique (ex: "2 + 2", "10 * 1.5")
        
    Returns:
        Résultat du calcul
    """
    import ast
    import operator
    
    # Nettoyer l'expression
    expression = expression.strip()
    
    # Sécurité: autoriser seulement chiffres et opérateurs de base
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Expression contient des caractères non autorisés")
    
    # Opérateurs autorisés
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }
    
    def eval_expr(node):
        """Évaluer de manière récursive et sécurisée"""
        if isinstance(node, ast.Num):  # Nombre
            return node.n
        elif isinstance(node, ast.BinOp):  # Opération binaire
            left = eval_expr(node.left)
            right = eval_expr(node.right)
            return ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Opération unaire
            return ops[type(node.op)](eval_expr(node.operand))
        else:
            raise ValueError(f"Opération non autorisée: {type(node).__name__}")
    
    try:
        # Parser et évaluer l'expression de manière sécurisée
        node = ast.parse(expression, mode='eval')
        result = eval_expr(node.body)
        return round(result, 2)
    except Exception as e:
        raise ValueError(f"Expression invalide: {e}")


# ============================================================
# TOOL 3: GET CURRENT TIME
# ============================================================

def get_current_time() -> str:
    """
    Obtenir la date et l'heure actuelles
    
    Returns:
        Date et heure formatées
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# TOOL 4: STRING MANIPULATION
# ============================================================

def string_length(text: str) -> int:
    """
    Calculer la longueur d'une chaîne
    
    Args:
        text: Texte à mesurer
        
    Returns:
        Nombre de caractères
    """
    return len(text)


# ============================================================
# TOOL 5: CURRENCY CONVERTER (Stub)
# ============================================================

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convertir une devise (taux simulés)
    
    Args:
        amount: Montant à convertir
        from_currency: Devise source (EUR, USD, GBP)
        to_currency: Devise cible
        
    Returns:
        Montant converti
    """
    # Taux de change simulés (EUR comme base)
    rates = {
        "EUR": 1.0,
        "USD": 1.10,
        "GBP": 0.85,
        "JPY": 156.0,
        "CHF": 0.95
    }
    
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    if from_currency not in rates or to_currency not in rates:
        raise ValueError(f"Devise non supportée")
    
    # Convertir via EUR
    amount_in_eur = amount / rates[from_currency]
    result = amount_in_eur * rates[to_currency]
    
    return round(result, 2)


# ============================================================
# FONCTION HELPER: GET ALL TOOLS
# ============================================================

def get_tools() -> Dict[str, Tool]:
    """
    Obtenir tous les outils disponibles
    
    Returns:
        Dictionnaire {nom: Tool}
    """
    tools = {
        "web_search": Tool(
            name="web_search",
            description="Recherche d'informations sur le web. Utilise cet outil pour trouver des faits, prix, données actuelles.",
            func=web_search,
            parameters={
                "query": "string - La requête de recherche"
            }
        ),
        
        "calculator": Tool(
            name="calculator",
            description="Calcule des expressions mathématiques. Supporte +, -, *, /, (), nombres décimaux.",
            func=calculator,
            parameters={
                "expression": "string - L'expression mathématique à évaluer (ex: '10 * 1.5 + 20')"
            }
        ),
        
        "get_current_time": Tool(
            name="get_current_time",
            description="Obtient la date et l'heure actuelles.",
            func=get_current_time,
            parameters={}
        ),
        
        "string_length": Tool(
            name="string_length",
            description="Calcule le nombre de caractères dans un texte.",
            func=string_length,
            parameters={
                "text": "string - Le texte à mesurer"
            }
        ),
        
        "convert_currency": Tool(
            name="convert_currency",
            description="Convertit un montant d'une devise à une autre. Supporte: EUR, USD, GBP, JPY, CHF.",
            func=convert_currency,
            parameters={
                "amount": "float - Le montant à convertir",
                "from_currency": "string - La devise source (ex: 'EUR')",
                "to_currency": "string - La devise cible (ex: 'USD')"
            }
        )
    }
    
    return tools


def get_tools_description() -> str:
    """
    Obtenir une description textuelle de tous les outils
    pour le prompt de l'agent
    
    Returns:
        Description formatée des outils
    """
    tools = get_tools()
    descriptions = []
    
    for tool_name, tool in tools.items():
        descriptions.append(tool.get_signature())
    
    return "\n".join(descriptions)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DES OUTILS")
    print("=" * 60)
    
    tools = get_tools()
    
    # Test web_search
    print("\n1. Test web_search:")
    result = tools["web_search"].execute(query="Paris Londres prix")
    print(f"   Résultat: {result}")
    
    # Test calculator
    print("\n2. Test calculator:")
    result = tools["calculator"].execute(expression="150 * 0.8")
    print(f"   Résultat: {result}")
    
    # Test get_current_time
    print("\n3. Test get_current_time:")
    result = tools["get_current_time"].execute()
    print(f"   Résultat: {result}")
    
    # Test string_length
    print("\n4. Test string_length:")
    result = tools["string_length"].execute(text="Hello World")
    print(f"   Résultat: {result}")
    
    # Test convert_currency
    print("\n5. Test convert_currency:")
    result = tools["convert_currency"].execute(amount=100, from_currency="EUR", to_currency="USD")
    print(f"   Résultat: {result}")
    
    # Afficher les descriptions
    print("\n" + "=" * 60)
    print("DESCRIPTIONS DES OUTILS (pour prompt)")
    print("=" * 60)
    print(get_tools_description())
