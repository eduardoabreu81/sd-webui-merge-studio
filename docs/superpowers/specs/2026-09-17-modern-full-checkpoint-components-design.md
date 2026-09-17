# Design: componentes modernos em Full Checkpoint

**Data:** 2026-09-17  
**Status:** aprovado para planejamento  
**Escopo:** Merge Studio executado dentro do Forge Neo

## 1. Contexto

O fluxo atual de **Full Checkpoint** consegue serializar os objetos que o Forge já montou no engine e oferece uma substituição explícita para VAE por meio de **Bake VAE**. Entretanto, não existe uma seleção equivalente e determinística para os text encoders exigidos pelas arquiteturas modernas.

Confiar implicitamente em `forge_additional_modules` não resolve o problema: o usuário não enxerga de forma inequívoca qual arquivo ocupou cada função, a composição pode depender do estado global do Forge e o checkpoint produzido pode não ser autossuficiente.

Este design transforma os componentes auxiliares em partes explícitas do Full Checkpoint. O Merge Studio continuará usando o loader e o `model_config` do Forge como autoridade final para montar e serializar a arquitetura.

O suporte atual do Merge Studio ao Anima 28/40/52 blocos é uma capacidade já entregue e não será reimplementado por este trabalho. O remapeamento entre gerações continuará consultando primeiro o `process_anima` do Forge e usando as tabelas locais somente como fallback. Esta entrega acrescenta a composição explícita de componentes e deve proteger esse suporte contra regressões.

## 2. Objetivo

Permitir que o usuário adicione ou substitua os text encoders e o VAE de um modelo moderno antes de gerar um Full Checkpoint autossuficiente, com:

- slots específicos por arquitetura;
- seleção explícita de arquivos locais;
- validação de função e compatibilidade;
- precisão controlada por componente;
- proveniência registrada;
- verificação de que o arquivo final reabre sem módulos externos.

## 3. Não objetivos

Esta entrega não pretende:

- interpolar, somar ou fazer merge dos pesos de text encoders ou VAE entre os modelos A, B e C;
- converter GGUF, Nunchaku, SVDQ, NF4 ou referências indiretas em tensores convencionais;
- aceitar qualquer formato diferente de `.safetensors` no novo seletor de componentes;
- criar slots modulares artificiais para SD1, SDXL, Illustrious, Pony, NoobAI ou Mugen;
- tratar PiD como arquitetura de checkpoint gerativo modular; PiD é um upscaler i2i e fica fora deste sistema;
- substituir o LLM Adapter do Anima, que pertence ao diffusion model e não é o Qwen3 usado como text encoder.
- recriar ou dividir o suporte Anima existente em arquiteturas rígidas separadas para 28, 40 e 52 blocos;
- tratar o adapter legado separado do Anima 3.8B v1 como text encoder ou componente modular.

## 4. Princípio central de composição

O pipeline terá duas fases claramente separadas:

1. O Merge Studio faz o merge somente do diffusion model de A, B e, quando aplicável, C.
2. Após o merge, os componentes selecionados são montados no resultado: text encoder(s) e VAE.

Os componentes externos nunca entram na matemática de merge A/B/C. Eles são incorporados integralmente, depois do diffusion merge, e serializados pelo `model_config` da arquitetura.

Um resultado só será anunciado como Full Checkpoint funcional depois de ser reaberto pelo `forge_loader` sem depender de `forge_additional_modules` ou de qualquer outro estado global.

## 5. Arquiteturas e slots

Os slots serão resolvidos por duas fontes complementares:

1. o perfil de capacidades exposto pelo `model_config` carregado pelo Forge, que é a autoridade de runtime;
2. uma política local pequena, usada somente para semântica que o Forge não expressa diretamente: rótulos, compatibilidade segura de assinaturas, estado de suporte e providers externos.

A matriz abaixo é a política conhecida para experiência imediata e testes. Ela não será uma cópia rígida da lista de nomes de classes do Forge. Uma futura arquitetura ou classe renomeada poderá ser descoberta genericamente quando expuser `clip_target`, prefixos, processadores de salvamento e layout de VAE suficientes; capacidades ambíguas falharão de forma contextual, sem adivinhação.

| Arquitetura | Text encoder(s) | VAE | Estado inicial |
|---|---|---|---|
| Flux 1 | CLIP-L + T5XXL | Flux AE | suportado |
| Flux 2 Klein 4B | Qwen3 4B | Flux 2 VAE | suportado |
| Flux 2 Klein 9B | Qwen3 8B | Flux 2 VAE | suportado |
| Wan 2.x | UMT5XXL | Wan VAE | suportado |
| Qwen-Image / Edit | Qwen2.5-VL 7B | Qwen Image VAE | suportado |
| Anima 28/40/52 | Qwen3 0.6B | Qwen Image VAE | merge existente preservado; composição suportada |
| Anima 3.8B v1.1 Semantic Connector v2 | Qwen3 0.6B + Qwen3.5 4B | Qwen Image VAE | overlay condicional ao provider de runtime |
| Krea2 | Qwen3-VL 4B | Qwen Image VAE | suportado |
| Z-Image | Qwen3 4B | Flux AE | suportado, com testes próprios |
| Lumina Image 2 | Gemma2 2B | Flux AE | suportado, com testes próprios |
| Chroma | T5XXL | VAE compatível com Flux | experimental |
| Ernie-Image | Ministral3 3B | Flux 2 VAE | experimental |

Os estados do registro serão:

- `supported`: composição modular habilitada normalmente;
- `experimental`: composição habilitada com aviso visível e exigência das mesmas validações finais;
- `not_applicable`: mantém o fluxo tradicional, sem slots modulares;
- `unknown`: tenta primeiro a descoberta de capacidades pelo Forge; bloqueia a composição modular apenas quando o contrato resultante for incompleto ou ambíguo.

## 6. Arquitetura interna

### 6.1 `forge_capabilities.py`

Camada de adaptação que lê a instância real de `engine.model_config`, sem depender do nome literal de sua classe, e normaliza:

- identidade e evidências da família vindas de `unet_config`, `huggingface_repo` e demais atributos declarados pelo Forge;
- alvos de text encoder em `clip_target`;
- `text_encoder_key_prefix` e `vae_key_prefix`;
- processadores `process_clip_state_dict_for_saving` e `process_vae_state_dict_for_saving`;
- `latent_format` e demais evidências de VAE;
- fingerprint diagnóstico das capacidades observadas.

O perfil será derivado depois do load/preflight e será a autoridade final. A inspeção de header continuará sendo rápida e provisória. Para uma arquitetura futura desconhecida pela política local, um perfil Forge completo poderá produzir slots genéricos; se faltar uma capacidade essencial, o sistema recusará a composição e informará exatamente o atributo ausente.

### 6.2 `component_registry.py`

Módulo puro e declarativo responsável pela política complementar:

- identificadores semânticos conhecidos e aliases estáveis;
- definição conhecida dos slots e overlays por arquitetura;
- obrigatoriedade e cardinalidade dos slots;
- assinaturas aceitas para cada função;
- famílias de VAE compatíveis;
- restrições de formato e armazenamento;
- nível de suporte e mensagens associadas.

O registro não carregará modelos, não dependerá da interface e não reproduzirá a lista de classes de `model_list.py`. Nomes de classes Forge não serão chaves de compatibilidade.

### 6.3 `component_providers.py`

Contrato pequeno para componentes cujo consumo pertence a uma extensão/runtime adicional. O primeiro provider será o overlay do Anima 3.8B v1.1:

- reconhecerá somente o bundle v2 por metadata e pelo prefixo estrutural `net.anima_v2_connector.`;
- manterá o Semantic Connector v2 dentro do diffusion model;
- acrescentará o slot Qwen3.5 4B ao perfil base Anima;
- recusará o adapter legado separado como componente;
- só declarará a saída validada quando o provider ativo conseguir consumir o Qwen3.5 incorporado, ou quando houver integração equivalente comprovada.

O core atual do Forge reconhece o Qwen3 0.6B e o VAE do Anima. O provider existe porque o runtime complementar do Anima 3.8B hoje procura o Qwen3.5 em arquivo externo; simplesmente gravar esses tensores no checkpoint sem adaptar o consumidor produziria um falso Full Checkpoint.

### 6.4 `component_bundle.py`

Serviço de composição responsável por:

- inspecionar os `.safetensors` selecionados;
- classificar cada arquivo por função;
- atribuir o arquivo ao slot correto;
- rejeitar slots ausentes, duplicados ou incompatíveis;
- resolver a precisão física de cada componente;
- construir explicitamente `additional_state_dicts` na ordem esperada pelo Forge;
- executar um preflight com o loader do Forge;
- produzir a proveniência usada pela receita e pela interface.

### 6.5 `checkpoint_merge.py`

Continuará responsável pela matemática do merge e pela gravação final. Passará a receber uma composição já validada, em vez de inferir silenciosamente os componentes a partir das opções globais do Forge.

O salvamento seguirá o caminho do `model_config.process_*_for_saving` da engine carregada, para que os namespaces finais sejam os definidos pela própria arquitetura no Forge.

### 6.6 Inspetores

O inspetor de componentes será corrigido antes de alimentar os seletores. A classificação não poderá concluir que um arquivo é VAE apenas porque contém prefixos genéricos como `encoder.` ou `decoder.`; assinaturas de T5, UMT5 e demais encoders terão precedência e critérios próprios.

O detector de arquitetura também será separado de heurísticas genéricas de nomes. Em particular, a presença de chaves Qwen não poderá, isoladamente, classificar qualquer modelo como Anima.

No runtime, a identificação Anima não poderá depender de `type(engine.model_config).__name__ == "Anima"`. Evidências estruturais do perfil Forge substituirão esse teste, enquanto a contagem 28/40/52 continuará sendo um detalhe de compatibilidade do diffusion model, não três arquiteturas novas.

## 7. Interface

Ao selecionar **Full Checkpoint**, a interface inspeciona o Modelo A e mostra uma seção de componentes apenas quando o perfil provisório ou autoritativo exigir composição modular.

Para famílias conhecidas, o header fornece imediatamente um perfil provisório. Se uma atualização do Forge introduzir uma classe ou arquitetura não conhecida localmente, a interface oferecerá a resolução pelo Forge e atualizará os slots a partir do perfil de capacidades retornado, sem exigir uma tabela nova apenas por causa do nome da classe.

As linhas serão renderizadas em quantidade variável conforme o perfil resolvido, sem assumir que três componentes continuará sendo o máximo. Cada linha de slot terá:

- nome funcional específico, como `CLIP-L`, `T5XXL` ou `Qwen3 0.6B`;
- seletor de arquivo compatível;
- opção `Embedded/current` somente quando o componente realmente existir no engine de origem;
- estado `Not selected — required` quando o componente obrigatório estiver ausente;
- seletor de precisão;
- resumo da precisão física e da classificação detectada;
- erro contextual quando o arquivo não puder ocupar aquele slot.

O atual **Bake VAE** será migrado para o slot explícito de VAE. O controle global **Text Encoder Format** será substituído pela precisão individual de cada text encoder.

Os módulos configurados globalmente no Forge poderão ser apresentados como sugestões locais, mas nunca serão adotados automaticamente. Toda inclusão deverá aparecer no formulário e na receita.

Arquiteturas `not_applicable` preservarão o comportamento tradicional e não receberão seletores que não façam sentido.

## 8. Validação de arquivos e compatibilidade

O novo seletor aceitará exclusivamente arquivos materializados `.safetensors`.

Serão recusados antes do carregamento:

- extensões diferentes de `.safetensors`;
- GGUF;
- Nunchaku e SVDQ;
- NF4;
- arquivos que sejam somente referência para pesos externos;
- componentes sem assinatura suficiente para uma função conhecida.

A classificação será específica, não apenas “text encoder”. Exemplos de funções distintas:

- CLIP-L e CLIP-G;
- T5XXL e UMT5XXL;
- Qwen3 por tamanho;
- Qwen3-VL;
- Qwen2.5-VL;
- Gemma2;
- Ministral3;
- VAE por layout latente e família.

Exemplos de incompatibilidades que devem ser detectadas cedo:

- Qwen3-VL 4B não substitui o Qwen3 0.6B do Anima;
- Qwen3.5 4B do Anima 3.8B v1.1 é um segundo encoder e não substitui o Qwen3 0.6B nativo;
- o adapter separado do Anima 3.8B v1 não ocupa nenhum slot de text encoder;
- T5XXL não substitui UMT5XXL;
- Flux 1 exige conjuntamente CLIP-L e T5XXL;
- um VAE com layout latente incompatível não ocupa o slot apenas por ser chamado de VAE;
- quantização Qwen2D só é aceita onde o caminho do Forge usado pela arquitetura consegue materializá-la para o salvamento pretendido.

Depois das validações estáticas, o `forge_loader` fará o preflight com exatamente os arquivos selecionados. Erros serão convertidos em mensagens que identifiquem arquitetura, slot, arquivo e motivo original.

O preflight reconciliará o perfil provisório do header com o perfil autoritativo derivado da engine. Mudanças futuras do Forge que removam ou alterem capacidades exigidas devem falhar com o fingerprint observado, nunca cair silenciosamente em uma tabela antiga.

## 9. Precisão por componente

Cada slot terá como padrão **Same as component source**.

Essa opção lê o header do próprio `.safetensors` que fornece o componente. Não reutiliza o dtype do Modelo A e não confia no dtype temporário que o Forge possa ter aplicado durante o carregamento.

Regras:

- tensores flutuantes preservam o dtype físico de origem por chave quando não houver conversão explícita;
- chaves ausentes no mapa de origem mantêm o dtype produzido pelo Forge;
- formatos explicitamente escolhidos só serão oferecidos quando forem materializáveis no Full Checkpoint normal;
- a precisão pode ser diferente entre diffusion model, cada text encoder e VAE;
- o header final será relido para conferir o dtype efetivamente gravado.

## 10. Proveniência e receita

A receita registrará, para cada componente:

- função/slot;
- nome do arquivo;
- hash;
- classificação detectada;
- origem (`embedded` ou arquivo explícito);
- precisão física de origem;
- precisão solicitada para saída;
- estado de suporte da arquitetura.

O registro não armazenará uma dependência oculta das opções globais de Additional Modules. O objetivo é permitir que outra execução reconstrua a mesma composição de maneira auditável.

## 11. Verificação do resultado

Após salvar, o Merge Studio executará obrigatoriamente:

1. leitura do header do arquivo final;
2. conferência dos namespaces esperados de diffusion model, text encoder(s) e VAE;
3. conferência dos dtypes e da presença de todos os slots obrigatórios;
4. reabertura pelo `forge_loader` com a lista de módulos externos vazia.

Quando um componente depender de provider adicional, como o Qwen3.5 do Anima 3.8B v1.1, haverá ainda uma quinta etapa: validação de que o provider ativo localizou e consumiu o componente incorporado. O reload do core Forge isoladamente não será aceito como prova suficiente desse caminho semântico.

Somente o sucesso das quatro etapas autoriza a mensagem de Full Checkpoint concluído.

Se a gravação terminar, mas a reabertura falhar, o arquivo será preservado e claramente marcado como **não validado**. A interface não afirmará que ele é um Full Checkpoint funcional.

## 12. Tratamento de erros

- Slot obrigatório ausente: desabilita a execução e informa qual componente falta.
- Arquivo incompatível: falha antes do merge e explica a assinatura encontrada e a esperada.
- Arquitetura desconhecida: tenta descoberta pelo perfil Forge; mantém o merge existente e bloqueia apenas se o perfil for insuficiente ou ambíguo.
- Provider obrigatório ausente: preserva o merge existente, mas não permite afirmar que o Full Checkpoint com aquele componente é autossuficiente.
- Arquitetura experimental: exige confirmação visual e mantém todas as validações.
- Falha no preflight: nenhum merge longo é iniciado.
- Falha no reload final: resultado preservado como não validado, com o erro do Forge disponível.

## 13. Estratégia de testes

### 13.1 Testes unitários

Usar headers e state dicts sintéticos pequenos para cobrir:

- detecção das arquiteturas registradas;
- classificação de cada tipo de encoder;
- distinção entre T5XXL, UMT5XXL e VAE com prefixos genéricos;
- distinção entre as variantes Qwen por família e tamanho;
- descoberta por capacidades que permanece estável quando a classe Forge é renomeada;
- descoberta genérica de uma futura configuração Forge completa, sem adicionar seu nome ao registro;
- falha segura quando `clip_target`, prefixos ou processadores de salvamento estiverem incompletos;
- Anima 28/40/52 como regressão do suporte atual, sem três registros de arquitetura;
- overlay v2 do Anima 3.8B e rejeição do adapter legado como text encoder;
- validação das famílias de VAE;
- slots obrigatórios, ausentes e duplicados;
- recusa de formatos e armazenamentos não suportados;
- preservação de precisão por componente e por chave;
- geração determinística da proveniência.

### 13.2 Testes de integração

Cobrir:

- merge do diffusion model sem interpolar componentes externos;
- construção de `additional_state_dicts` na ordem correta;
- preflight da engine com composição explícita;
- serialização dos namespaces pelo `model_config`;
- reload final sem Additional Modules globais;
- validação provider-aware do Qwen3.5 incorporado no Anima 3.8B v1.1;
- falha segura quando o arquivo é salvo, mas não pode ser reaberto.

### 13.3 Matriz de arquitetura

Flux 1, Flux 2 Klein 4B/9B, Wan 2.x, Qwen-Image/Edit, Anima, Krea2, Z-Image e Lumina Image 2 terão políticas conhecidas e casos próprios. Anima manterá testes de regressão 28/40/52 e terá um overlay específico para o bundle 3.8B v1.1. Chroma e Ernie-Image permanecerão experimentais até terem fixtures e validação real no runtime correspondente. Um fake model config futuro validará que a descoberta não depende dessa lista.

Testes com modelos completos, por causa do tamanho, poderão ser marcados como smoke tests manuais ou condicionais; isso não substituirá os testes automatizados com fixtures mínimas nem o reload obrigatório durante o uso real.

## 14. Migração e compatibilidade

A implementação será feita internamente nesta ordem:

1. contrato de capacidades Forge e política complementar;
2. correção dos inspetores;
3. composição e validação sem interface;
4. testes de precisão e proveniência;
5. migração de Bake VAE e Text Encoder Format para slots;
6. interface dinâmica;
7. validação pós-save e reload independente;
8. testes por arquitetura e documentação do usuário.

O fluxo atual continuará disponível para arquiteturas tradicionais. Receitas antigas que usem Bake VAE serão interpretadas pelo slot de VAE quando possível, sem inventar text encoders externos.

## 15. Critérios de aceite

O recurso estará pronto quando:

- o Modelo A determinar de forma confiável os slots necessários;
- uma classe Forge renomeada ou futura com o mesmo contrato funcional não exigir novo hardcode por nome;
- somente `.safetensors` compatíveis puderem ocupar cada slot;
- nenhum componente selecionado participar da interpolação A/B/C;
- o usuário puder substituir separadamente cada encoder e o VAE;
- `Same as component source` preservar a precisão física de cada origem;
- a receita listar todos os componentes e hashes usados;
- a saída contiver todos os namespaces obrigatórios;
- a saída reabrir no Forge sem módulos externos;
- componentes dependentes de provider externo só serem declarados válidos após consumo comprovado pelo provider;
- o merge Anima 28/40/52 já existente permanecer coberto por regressão;
- arquiteturas tradicionais continuarem funcionando sem regressão;
- PiD não apareça no registro de composição modular;
- Chroma e Ernie permaneçam claramente experimentais até validação real.
