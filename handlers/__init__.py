from handlers import adain_handler_balanced

DEFAULT_SA_ARGS = adain_handler_balanced.StyleAlignedArgs(share_group_norm=False,
                                                            share_layer_norm=False,
                                                            share_attention=True,
                                                            adain_queries=True,
                                                            adain_keys=True,
                                                            adain_values=False)

FALSE_SA_ARGS = adain_handler_balanced.StyleAlignedArgs(share_group_norm=False,
                                                            share_layer_norm=False,
                                                            share_attention=False,
                                                            adain_queries=False,
                                                            adain_keys=False,
                                                            adain_values=False)
        
def get_handler():
    return adain_handler_balanced.Handler