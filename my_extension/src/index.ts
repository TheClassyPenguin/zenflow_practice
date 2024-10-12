import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ISettingRegistry } from '@jupyterlab/settingregistry';

/**
 * Initialization data for the tracking-test extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'tracking-test:plugin',
  description: 'A JupyterLab extension that gets extra telemetry data',
  autoStart: true,
  optional: [ISettingRegistry],
  activate: (app: JupyterFrontEnd, settingRegistry: ISettingRegistry | null) => {
    console.log('JupyterLab extension tracking-test is activated!');

    if (settingRegistry) {
      settingRegistry
        .load(plugin.id)
        .then(settings => {
          console.log('tracking-test settings loaded:', settings.composite);
        })
        .catch(reason => {
          console.error('Failed to load settings for tracking-test.', reason);
        });
    }
  }
};

export default plugin;
