# Design: componentes modernos em AIO (Full Checkpoint)

**Data:** 2026-09-17  
**Status:** revisado contra o código do Forge Neo  
**Escopo:** Merge Studio executado dentro do Forge Neo

> **Revisão de 2026-09-17.** Este design foi auditado contra o fonte do Forge Neo
> (`backend/loader.py`, `huggingface_guess/model_list.py`) e contra a biblioteca de
> modelos instalada. Sete decisões mudaram o desenho:
>
> | | Decisão |
> |---|---|
> | D1 | Text encoder quantizado é aceito em passthrough bit-exato — `fp8_scaled` é o formato padrão distribuído, não exceção |
> | D2 | O Qwen3.5 4B do Anima 3.8B sai do escopo: o Forge descarta as chaves, e não há evidência estrutural que decida qual encoder o modelo quer |
> | D3 | Anima tem uma política só, para todas as gerações |
> | D4 | Os slots vêm do `clip_target` do Forge, não de matriz copiada |
> | D5 | O inspector reporta campos reais por arquitetura, não `clip`/`vae` genéricos |
> | D6 | Modelo AIO: duas opções por linha, sem auto-preenchimento, bloqueio se incompleto |
> | D7 | Seguir as regras do Forge; checkpoint com namespace errado é problema de quem gerou |
>
> **Nomenclatura:** o resultado autossuficiente passa a se chamar **AIO**, termo já usado
> pela comunidade. "Full Checkpoint tradicional" fica reservado para SD15/SDXL, onde os
> encoders são sempre embutidos.

## 1. Contexto

O fluxo atual de **Full Checkpoint** consegue serializar os objetos que o Forge já montou no engine e oferece uma substituição explícita para VAE por meio de **Bake VAE**. Entretanto, não existe uma seleção equivalente e determinística para os text encoders exigidos pelas arquiteturas modernas.

Confiar implicitamente em `forge_additional_modules` não resolve o problema: o usuário não enxerga de forma inequívoca qual arquivo ocupou cada função, a composição pode depender do estado global do Forge e o checkpoint produzido pode não ser autossuficiente.

Este design transforma os componentes auxiliares em partes explícitas do AIO. O Merge Studio continuará usando o loader e o `model_config` do Forge como autoridade final para montar e serializar a arquitetura.

O suporte atual do Merge Studio ao Anima 28/40/52 blocos é uma capacidade já entregue e não será reimplementado por este trabalho. O remapeamento entre gerações continuará consultando primeiro o `process_anima` do Forge e usando as tabelas locais somente como fallback. Esta entrega acrescenta a composição explícita de componentes e deve proteger esse suporte contra regressões.

## 2. Objetivo

Permitir que o usuário adicione ou substitua os text encoders e o VAE de um modelo moderno antes de gerar um AIO autossuficiente, com:

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
- desquantizar um componente escalado para oferecer conversão explícita de precisão (§9.1);
- aceitar qualquer formato diferente de `.safetensors` no novo seletor de componentes;
- criar slots de text encoder para SD1, SDXL, Illustrious, Pony, NoobAI ou Mugen, cujos encoders são sempre embutidos — o slot de VAE continua existindo para eles;
- tratar PiD como arquitetura de checkpoint gerativo; PiD é um upscaler i2i e fica fora deste sistema, identificado por `latent_format = RGB`;
- substituir o LLM Adapter do Anima, que pertence ao diffusion model e não é o Qwen3 usado como text encoder;
- recriar ou dividir o suporte Anima existente em arquiteturas rígidas separadas para 28, 40 e 52 blocos;
- criar slot, overlay ou provider para o Qwen3.5 4B ou para o adapter legado do Anima 3.8B (D2);
- adivinhar qual arquivo instalado preenche qual slot (D6);
- corrigir checkpoints que embutem componentes sob namespace que sua arquitetura não declara (D7).

## 4. Princípio central de composição

O pipeline terá duas fases claramente separadas:

1. O Merge Studio faz o merge somente do diffusion model de A, B e, quando aplicável, C.
2. Após o merge, os componentes selecionados são montados no resultado: text encoder(s) e VAE.

Os componentes externos nunca entram na matemática de merge A/B/C. Eles são incorporados integralmente, depois do diffusion merge, e serializados pelo `model_config` da arquitetura.

Um resultado só será anunciado como AIO funcional depois de ser reaberto pelo `forge_loader` sem depender de `forge_additional_modules` ou de qualquer outro estado global.

## 5. Arquiteturas e slots

**A lista de slots é do Forge, não nossa (D4).** O `clip_target` de cada `model_config`
declara exatamente quais text encoders a arquitetura consome, e o `vae_key_prefix` declara
o VAE. Replicar isso numa matriz local seria o hardcode que este design existe para evitar:
uma arquitetura nova no Forge passaria a aparecer sozinha, sem edição de registro.

A política local fica apenas com o que o Forge não expressa:

- rótulo legível de cada slot;
- assinaturas aceitas por função;
- formatos de armazenamento aceitos (§8);
- estado de suporte.

Não existem dois mundos separados de "modular" e "tradicional". **Toda arquitetura tem um
conjunto obrigatório**; o que varia é o tamanho dele e se os componentes já vêm embutidos.
SD15 e SDXL têm slot de VAE — o que eles não têm é encoder selecionável, porque
`cond_stage_model.` já vem no arquivo.

A tabela abaixo é **derivada** do Forge e serve como referência de usuário e base de teste,
não como fonte de verdade em runtime:

| Arquitetura | Text encoders | VAE |
|---|---|---|
| **Flux 1** — Dev / Schnell / Kontext | **2** — `clip_l` + `t5xxl` | ae |
| SDXL / Pony / Illustrious / NoobAI | 2 — `clip_l` + `clip_g`, sempre embutidos | sdxl-vae |
| SD 1.5 | 1 — `clip_l`, sempre embutido | vae-ft-mse |
| Anima (todas as gerações) | 1 — `qwen3_06b` | Qwen-Image VAE |
| Krea2 | 1 — `qwen3vl_4b` | Qwen-Image VAE |
| Qwen-Image / Edit | 1 — `qwen25_7b` | Qwen-Image VAE |
| Z-Image / Turbo | 1 — `qwen3_4b` | ae (Flux) |
| Flux.2-Klein 4B | 1 — `qwen3_4b` | flux2-vae |
| Flux.2-Klein 9B | 1 — `qwen3_8b` | flux2-vae |
| Wan 2.x | 1 — `umt5xxl` | wan vae |
| Lumina Image 2 | 1 — `gemma2_2b` | ae (Flux) |
| Chroma | 1 — `t5xxl` | ae (Flux) |
| Ernie-Image | 1 — `ministral3_3b` | flux2-vae |

A regra geral é **1 text encoder + 1 VAE**; o **Flux 1** é a única exceção, com 2 encoders.
Flux 2 Klein, apesar do nome, tem 1 — e herda da mesma classe `Flux` do Forge,
sobrescrevendo o `clip_target`. Chroma faz o mesmo herdando de `FluxSchnell`. A hierarquia
de classes não diz quantos encoders uma arquitetura consome; só o `clip_target` diz.
Flux Kontext não é classe separada no Forge — usa a mesma `Flux` e é distinguido por
`"kontext"` no nome do arquivo, por isso herda os mesmos dois encoders.

**Anima é uma entrada só (D3).** As gerações de 28, 40 e 52 blocos compartilham
`qwen3_06b` + VAE Qwen-Image. Não há overlay de Qwen3.5 4B: o `clip_target` do Forge
declara apenas `qwen3_06b`, e tudo que não casa com um alvo declarado é descartado no
`split_state_dict`. Além disso, existem checkpoints de 52 blocos cujos autores declaram
explicitamente usar só o encoder nativo — contagem de blocos não determina o encoder.

**PiD fica fora** por capacidade, não por nome: é a única classe com `latent_format = RGB`,
coerente com ser um upscaler i2i.

Os estados do registro serão:

- `supported`: composição habilitada normalmente;
- `experimental`: composição habilitada com aviso visível e as mesmas validações finais;
- `unknown`: tenta a descoberta de capacidades pelo Forge; bloqueia apenas quando o contrato
  resultante for incompleto ou ambíguo.

## 6. Arquitetura interna

### 6.1 `forge_capabilities.py`

Camada de adaptação que lê a instância real de `engine.model_config`, sem depender do nome literal de sua classe, e normaliza:

- identidade e evidências da família vindas de `unet_config`, `huggingface_repo` e demais atributos declarados pelo Forge;
- alvos de text encoder em `clip_target`;
- `text_encoder_key_prefix` e `vae_key_prefix`;
- `latent_format` e demais evidências de VAE;
- fingerprint diagnóstico das capacidades observadas.

Três detalhes medidos no fonte do Forge que a implementação precisa respeitar:

1. **`clip_target` é método antes do load e dict depois.** `loader.py` faz
   `guess.clip_target = guess.clip_target(sd)` durante `split_state_dict`. As duas formas
   têm de ser normalizadas.
2. **`clip_target` é condicional ao state_dict** em `Flux`, `Chroma`, `Lumina2` e
   `QwenImage`, e é avaliado **depois** de os módulos adicionais serem mesclados. Se um
   componente não for fornecido, o Forge simplesmente **não declara o slot** — ele não
   reclama. Ausência é detectada comparando com a política, nunca esperando erro do Forge.
3. **Os `process_*_state_dict_for_saving` existem para todas as arquiteturas**, herdados da
   `BASE`, e apenas aplicam `text_encoder_key_prefix[0]` / `vae_key_prefix[0]`. Só as
   legadas e o Chroma fazem override. Portanto a presença deles **não** é uma capacidade
   discriminante e não deve virar condição de falha.

O perfil será derivado depois do load/preflight e será a autoridade final. A inspeção de header continuará sendo rápida e provisória. Para uma arquitetura futura desconhecida pela política local, um perfil Forge completo poderá produzir slots genéricos; se faltar uma capacidade essencial, o sistema recusará a composição e informará exatamente o atributo ausente.

Vale registrar que o próprio Forge usa `huggingface_repo` — uma string — como chave de
compatibilidade (`if "Anima" in guess.huggingface_repo`), e chega a usar o nome do arquivo
para variantes como Kontext. Alinhar a identidade a `huggingface_repo` e
`unet_config["image_model"]` é o mais resistente a atualização: a extensão passa a quebrar
exatamente quando o Forge quebra, nunca antes e nunca em silêncio.

### 6.2 `component_registry.py`

Módulo puro e declarativo responsável **apenas** pela política que o Forge não expressa:

- rótulo legível de cada slot;
- assinaturas aceitas para cada função;
- formatos de armazenamento aceitos por slot (§8);
- famílias de VAE compatíveis;
- nível de suporte e mensagens associadas.

**O registro não define quais slots existem.** Isso vem do `clip_target` e do
`vae_key_prefix` do Forge (§5, D4). Não há matriz de arquiteturas, não há entradas por
arquitetura para obrigatoriedade e cardinalidade, e não há overlays.

O registro não carregará modelos, não dependerá da interface e não reproduzirá a lista de classes de `model_list.py`. Nomes de classes Forge não serão chaves de compatibilidade.

### 6.3 Providers externos — fora do escopo (D2)

A versão anterior deste design previa um `component_providers.py` cujo primeiro caso seria
o Qwen3.5 4B do Anima 3.8B v1.1. **Isso saiu**, por dois motivos independentes:

1. **O Forge descarta as chaves.** `Anima.clip_target` declara apenas
   `{"qwen3_06b.transformer": "text_encoder"}`. Em `split_state_dict`, tudo que não casa
   com um alvo declarado cai em `state_dict["ignore"]`, que é deletado logo em seguida.
   Embutir o Qwen3.5 somaria cerca de 4,8 GB ao arquivo sem qualquer efeito.

2. **Não há evidência estrutural que decida o encoder.** Existem checkpoints de 52 blocos
   cujos autores declaram usar somente o encoder nativo. Contagem de blocos não determina o
   encoder, e a presença do `net.anima_v2_connector.` tampouco — o Anima-3.8B v1 sequer
   tem o connector.

O Semantic Connector v2 continua sendo parte do diffusion model em qualquer caso, e o
adapter legado separado nunca ocupa slot. Nenhum módulo novo é criado para isso.

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

A discriminação de VAE não pode se apoiar em `decoder.conv_in.weight`: os VAE das famílias
Qwen-Image, Wan e Anima usam convolução 3D e **não têm essa chave** (é
`decoder.conv_in.conv.weight`). A hierarquia que de fato funciona:

```
conv 3D / temporal        -> família Qwen-Image / Wan / Anima   (z=16)
conv 2D + conv_in 16ch    -> família Flux AE
conv 2D + conv_in  4ch    -> SD / SDXL
```

Nomes de arquivo não são evidência de nada. Um encoder da biblioteca real chama-se
`animaLLMLayerwiseFP8_v1.safetensors` e está inteiramente em F16.

O detector de arquitetura também será separado de heurísticas genéricas de nomes. Em particular, a presença de chaves Qwen não poderá, isoladamente, classificar qualquer modelo como Anima.

**O relatório de componentes passa a ser por arquitetura (D5).** Hoje o inspector devolve
quatro campos fixos — `unet`, `clip`, `vae`, `llm_adapter` — cegos à arquitetura; para um
Anima, `clip: True` significa "tem um Qwen3 0.6B", e chamar isso de CLIP é falso. O
inspector identificará a arquitetura primeiro e só então reportará os campos reais.

Para cada componente embutido, três fatos distintos, nenhum escondido:

| fato | exemplo |
|---|---|
| qual componente | `qwen3_06b` (text encoder) |
| sob qual namespace | `cond_stage_model.` |
| quem lê esse namespace | Anima espera `text_encoders.` → o Forge não lê |

O terceiro é aviso, não omissão: o arquivo pode ser levado para outro runtime, e lá o
namespace pode ser lido. A ferramenta reporta o arquivo, não a opinião de um runtime.

No runtime, a identificação Anima não poderá depender de `type(engine.model_config).__name__ == "Anima"`. Evidências estruturais do perfil Forge substituirão esse teste, enquanto a contagem 28/40/52 continuará sendo um detalhe de compatibilidade do diffusion model, não três arquiteturas novas.

Os dois usos remanescentes de nome de classe como chave de compatibilidade —
`checkpoint_merge.py` e `lora_bake.py`, ambos testando `"SDXL" in type(...).__name__` para
escolher prefixo de VAE — caem pela mesma regra.

## 7. Interface

Ao selecionar **AIO**, a interface mostra uma linha por componente que a arquitetura do
Modelo A declara (§5).

### 7.1 Como funciona, na prática

Um checkpoint Anima que é só diffusion model — 190 dos 222 da biblioteca de referência:

```
Text Encoder   (•) [ escolher arquivo ▾ ]     <- lista de models/text_encoder/
VAE            (•) [ escolher arquivo ▾ ]     <- lista de models/VAE/
```

Duas linhas porque o Forge declara que Anima consome 1 encoder + 1 VAE. Para Flux seriam
três (`clip_l`, `t5xxl`, VAE). Para SDXL seria uma só (VAE), porque os encoders já vêm
dentro do arquivo.

Um checkpoint que **já traz** os componentes carregados:

```
Text Encoder   (•) Manter o que está no arquivo
               ( ) [ escolher arquivo ▾ ]
VAE            (•) Manter o que está no arquivo
               ( ) [ escolher arquivo ▾ ]
```

Se o usuário deixar uma linha em branco e mandar processar:

```
Select a text encoder to save as AIO — or switch to UNet only.
Select a VAE to save as AIO — or switch to UNet only.
Select a text encoder and a VAE to save as AIO — or switch to UNet only.
```

Para Flux, a mensagem nomeia qual dos dois encoders falta.

### 7.2 As quatro regras (D6)

1. **Quantas linhas** — decidido pelo `clip_target` do Forge, nunca por tabela local.
2. **O que cada linha oferece** — manter o que está no arquivo (apenas quando a engine
   realmente carregou o componente) ou escolher um arquivo da pasta correspondente.
3. **Não se adivinha** — nenhum auto-preenchimento a partir da varredura da pasta, e nenhuma
   adoção automática dos módulos globais do Forge. Quem quer embutir tem o conhecimento
   mínimo de TE e VAE; a tabela de §5 documenta o que cada arquitetura precisa.
4. **AIO incompleto trava** — todo slot declarado precisa de seleção. A execução é bloqueada
   com a mensagem nomeando o que falta e oferecendo UNet only como saída.

**São duas opções por linha no modo AIO, não três.** `Nenhum` deixa de existir aqui: "AIO
sem encoder" é contraditório. Quem quer um arquivo sem VAE usa **UNet only**, que já faz
esse trabalho. Isso remove o `bake_vae: "none"` do caminho AIO.

### 7.3 Detalhes de cada linha

- nome funcional específico, como `CLIP-L`, `T5XXL` ou `Qwen3 0.6B`;
- seletor de arquivo restrito aos compatíveis com aquele slot;
- seletor de precisão (§9);
- resumo da precisão física e da classificação detectada;
- erro contextual quando o arquivo não puder ocupar aquele slot.

O atual **Bake VAE** será migrado para o slot explícito de VAE — é literalmente o mesmo
controle, agora também disponível para o text encoder. O controle global
**Text Encoder Format** será substituído pela precisão individual de cada text encoder.

A opção "manter o que está no arquivo" é decidida pelo que a **engine carregou**, nunca pelo
header (D7). Um checkpoint que embute componentes sob namespace que sua arquitetura não
declara simplesmente não oferece a opção, e o usuário seleciona arquivos — comportamento
correto sem código especial. O inspector registra o fato como aviso (§6.6), porque o usuário
merece saber por que o "embutido" dele não apareceu.

O número máximo real de linhas é **três** (Flux). A interface não precisa assumir um teto
fixo, mas também não precisa de renderização dinâmica arbitrária para atender o escopo.

## 8. Validação de arquivos e compatibilidade

O novo seletor aceitará exclusivamente arquivos materializados `.safetensors`.

### 8.1 Formatos de armazenamento (D1)

**Encoder quantizado é aceito.** `fp8_scaled` é o formato **padrão** de distribuição no
Forge Neo, não uma exceção: a wiki oficial o lista para Flux, Z-Image, Wan 2.2, Qwen-Image,
Ernie-Image, Krea2 e PiD, e `fp8mixed` para Flux.2-Klein. Recusá-lo seria recusar o arquivo
recomendado da maioria das arquiteturas.

Aceitos, declarados por slot no registro:

- `plain` — FP16, BF16, FP32;
- `fp8_scaled` — pesos F8_E4M3 com tensores `weight_scale` em F32 e blocos U8;
- `fp8_mixed` — variante com `weight_scale_2`.

Recusados antes do carregamento:

- extensões diferentes de `.safetensors`;
- GGUF;
- Nunchaku e SVDQ;
- NF4;
- arquivos que sejam somente referência para pesos externos;
- componentes sem assinatura suficiente para uma função conhecida.

Anima é a única família cujo encoder oficial é apenas BF16, então o caminho mais usado hoje
não exercita o passthrough — ele existe para Krea2, Z-Image e as demais.

### 8.2 Classificação por função

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
- nem o Qwen3.5 4B nem o adapter separado do Anima 3.8B ocupam slot de text encoder (D2);
- T5XXL não substitui UMT5XXL;
- Flux 1 exige conjuntamente CLIP-L e T5XXL; Flux 2 Klein não, apesar do nome parecido;
- um VAE com layout latente incompatível não ocupa o slot apenas por ser chamado de VAE.

Um caso que exige cuidado: o encoder do **Krea2** (`qwen3vl_4b`) e o do **Z-Image**
(`qwen3_4b`) são dimensionalmente **idênticos** — 36 camadas, hidden 2560, vocab 151936,
32 heads, kv 8, head_dim 128, intermediate 9728. A única diferença estrutural é a torre de
visão (`model.visual.*`) do Qwen3-VL. Esse é o único discriminador disponível.

Nota sobre **Qwen2D**: não é uma quantização, é uma variante do VAE Qwen-Image. A wiki do
Forge o lista como opção de VAE (`qwen_image_vae, Qwen2D_VAE`), ao lado do padrão.

Depois das validações estáticas, o `forge_loader` fará o preflight com exatamente os arquivos selecionados. Erros serão convertidos em mensagens que identifiquem arquitetura, slot, arquivo e motivo original.

O preflight reconciliará o perfil provisório do header com o perfil autoritativo derivado da engine. Mudanças futuras do Forge que removam ou alterem capacidades exigidas devem falhar com o fingerprint observado, nunca cair silenciosamente em uma tabela antiga.

## 9. Precisão por componente

Cada slot terá como padrão **Same as component source**.

Essa opção lê o header do próprio `.safetensors` que fornece o componente. Não reutiliza o dtype do Modelo A e não confia no dtype temporário que o Forge possa ter aplicado durante o carregamento.

Regras:

- tensores flutuantes preservam o dtype físico de origem por chave quando não houver conversão explícita;
- chaves ausentes no mapa de origem mantêm o dtype produzido pelo Forge;
- formatos explicitamente escolhidos só serão oferecidos quando forem materializáveis no AIO;
- a precisão pode ser diferente entre diffusion model, cada text encoder e VAE;
- o header final será relido para conferir o dtype efetivamente gravado.

### 9.1 Componente quantizado: passthrough bit-exato (D1)

Para um componente em `fp8_scaled` ou `fp8_mixed`, `Same as component source` significa
**cópia bit-exata**, não "casar dtype por chave":

- zero conversão, inclusive nos pesos F8_E4M3;
- `weight_scale`, `weight_scale_2` e blocos U8 **nunca** são tocados, em nenhum modo;
- `fp16`, `bf16` e `fp32` explícitos **não são oferecidos** para esses componentes —
  converter exigiria desquantizar, o que está fora do escopo. A linha mostra apenas
  `Same as component source`.

Fica registrado como verificação pendente, não como premissa: **se o caminho de salvamento
do Forge preserva os `weight_scale` ao serializar um encoder escalado dentro do AIO.** Isso
só se responde com o Forge em execução.

### 9.2 O LLM Adapter do Anima

`process_anima` do Forge **move** as chaves `llm_adapter` do transformer **para dentro do
text encoder** durante o load. O código atual já compensa isso na gravação, devolvendo-as a
`model.diffusion_model.*` com o dtype do DiT.

Consequência para a composição: quando o usuário troca o encoder por um arquivo externo, as
chaves de `llm_adapter` presentes no balde do encoder vieram do **Modelo A**, não do arquivo
selecionado. `Same as component source` do slot de encoder não pode tentar casá-las contra o
header do arquivo externo.

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
3. conferência dos dtypes e da presença de todos os slots declarados;
4. reabertura pelo `forge_loader` com a lista de módulos externos vazia.

Os namespaces esperados do passo 2 vêm do **perfil resolvido** — `text_encoder_key_prefix[0]`
e `vae_key_prefix[0]` da engine carregada — nunca de tabela local.

Somente o sucesso das quatro etapas autoriza a mensagem de AIO concluído.

Se a gravação terminar, mas a reabertura falhar, o arquivo será preservado e claramente marcado como **não validado**. A interface não afirmará que ele é um AIO funcional.

O passo 4 é a exigência mais cara deste design, e é o que o justifica: na biblioteca de
referência, **apenas 2 de 222 checkpoints** embutem componentes sob os namespaces que a sua
própria arquitetura declara. Trinta usam namespace de outra família e dependem, na prática,
dos módulos globais. Reabrir com a lista vazia é o único jeito de distinguir um AIO de
verdade de um que só parece.

## 12. Tratamento de erros

- Slot sem seleção no modo AIO: bloqueia a execução, nomeia o que falta e oferece UNet only como saída (§7.1).
- Arquivo incompatível: falha antes do merge e explica a assinatura encontrada e a esperada.
- Arquitetura desconhecida: tenta descoberta pelo perfil Forge; mantém o merge existente e bloqueia apenas se o perfil for insuficiente ou ambíguo.
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
- `clip_target` nas duas formas — método e dict já resolvido;
- `clip_target` condicional que deixa de declarar um slot quando o componente não é fornecido;
- falha segura quando `clip_target` ou prefixos estiverem incompletos;
- Anima 28/40/52 como regressão do suporte atual, sem três registros de arquitetura;
- rejeição do Qwen3.5 4B e do adapter legado como text encoder do Anima;
- distinção entre `qwen3vl_4b` e `qwen3_4b`, idênticos exceto pela torre de visão;
- famílias de VAE pela hierarquia conv 3D / conv 2D 16ch / conv 2D 4ch, não por `decoder.conv_in.weight`;
- slots sem seleção no modo AIO;
- aceitação de `fp8_scaled` e `fp8_mixed`; recusa de GGUF, Nunchaku/SVDQ e NF4;
- passthrough bit-exato de componente escalado, com `weight_scale` e blocos U8 intactos;
- nome de arquivo não sendo evidência de formato nem de função;
- preservação de precisão por componente e por chave;
- geração determinística da proveniência.

### 13.2 Testes de integração

Cobrir:

- merge do diffusion model sem interpolar componentes externos;
- construção de `additional_state_dicts` na ordem correta;
- preflight da engine com composição explícita;
- serialização nos namespaces declarados pelo perfil resolvido;
- reload final sem Additional Modules globais;
- falha segura quando o arquivo é salvo, mas não pode ser reaberto.

### 13.3 Matriz de arquitetura

A cobertura de arquiteturas é feita com fake model configs que expõem o contrato funcional
do Forge, não com uma lista de nomes. Um config futuro e desconhecido deve produzir slots
utilizáveis sem edição de registro — esse é o teste central do design.

Anima mantém testes de regressão 28/40/52 e uma política única, sem overlay. Chroma e
Ernie-Image permanecem experimentais até terem fixtures e validação real no runtime
correspondente.

Fica registrado que a biblioteca de referência disponível é essencialmente Anima: 222 de 222
checkpoints. Testes de ponta a ponta com o Forge rodando só podem ser feitos nessa família;
as demais são cobertas por fixtures sintéticas. **Nenhuma arquitetura deve ser declarada
validada em runtime sem ter sido de fato carregada e reaberta.**

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

- os slots virem do `clip_target` do Forge, sem matriz de arquiteturas no registro;
- uma classe Forge renomeada ou futura com o mesmo contrato funcional não exigir novo hardcode por nome;
- somente `.safetensors` compatíveis puderem ocupar cada slot;
- `fp8_scaled` e `fp8_mixed` serem aceitos em passthrough, com `weight_scale` e blocos U8 intactos;
- nenhum componente selecionado participar da interpolação A/B/C;
- o usuário puder substituir separadamente cada encoder e o VAE, pelo mesmo controle que o Bake VAE já oferece;
- nenhum slot ser preenchido automaticamente, e nenhum módulo global ser adotado em silêncio;
- o modo AIO bloquear com mensagem acionável quando faltar seleção, oferecendo UNet only;
- `Same as component source` preservar a precisão física de cada origem;
- a receita listar todos os componentes e hashes usados;
- a saída contiver todos os namespaces declarados pelo perfil resolvido;
- a saída reabrir no Forge sem módulos externos;
- o inspector reportar componentes por arquitetura, com namespace e legibilidade pelo runtime;
- o merge Anima 28/40/52 já existente permanecer coberto por regressão;
- arquiteturas tradicionais continuarem funcionando sem regressão, com seu slot de VAE;
- PiD ficar fora por `latent_format = RGB`, não por nome;
- Chroma e Ernie permaneçam claramente experimentais até validação real.
