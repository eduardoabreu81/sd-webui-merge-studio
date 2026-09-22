"""Merge Studio's own modules.

These used to sit at the extension root, which is where `scripts/` puts them on
`sys.path` so the UI can import them. That made twenty generic names --
`lora_merge`, `merge_modes`, `quant_utils`, `component_ui` -- top-level modules
of the whole WebUI process. Another extension shipping a file of the same name
and doing the same thing would win or lose the slot in `sys.modules` by load
order, and the loser would silently import the wrong module.

One package, one name. The root now holds only what a Forge extension is
supposed to hold, and the collision surface is this directory's name.

Nothing is re-exported here on purpose: importing a submodule should cost only
that submodule, since `checkpoint_merge` pulls in torch and Forge's backend
while `merge_modes` deliberately does not.
"""
