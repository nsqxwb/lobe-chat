import { Markdown, MarkdownProps } from '@lobehub/ui';
import { memo } from 'react';
import { Flexbox } from 'react-layout-kit';

import ImageFileListViewer from '@/features/Conversation/Messages/User/ImageFileListViewer';
import { normalizeThinkTags, processWithArtifact } from '@/features/Conversation/utils';
import { AssistantContentBlock } from '@/types/message';

import Tool from './Tool';

interface AssistantBlockProps extends AssistantContentBlock {
  index: number;
  markdownProps?: Omit<MarkdownProps, 'className' | 'style' | 'children'>;
}
export const AssistantBlock = memo<AssistantBlockProps>(
  ({ id, tools, content, imageList, markdownProps }) => {
    // const generating = useChatStore(chatSelectors.isMessageGenerating(id));

    // const isToolCallGenerating = generating && (content === LOADING_FLAT || !content) && !!tools;
    const message = normalizeThinkTags(processWithArtifact(content));

    const showImageItems = !!imageList && imageList.length > 0;

    if (tools && tools.length > 0)
      return (
        <Flexbox gap={8}>
          {tools.map((toolCall, index) => (
            <Tool
              apiName={toolCall.apiName}
              arguments={toolCall.arguments}
              id={toolCall.id}
              identifier={toolCall.identifier}
              index={index}
              key={toolCall.id}
              messageId={id}
              payload={toolCall}
              type={toolCall.type}
            />
          ))}
        </Flexbox>
      );

    return (
      <Flexbox gap={8} id={id}>
        <Markdown {...markdownProps} variant={'chat'}>
          {message}
        </Markdown>
        {showImageItems && <ImageFileListViewer items={imageList} />}
      </Flexbox>
    );
  },
);
